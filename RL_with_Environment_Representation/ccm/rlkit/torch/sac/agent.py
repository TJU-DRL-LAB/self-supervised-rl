import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(nn.Module):#context encoder -> action output (during training and sampling)

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 context_encoder_target,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.W = nn.Parameter(torch.rand(latent_dim, latent_dim))
        self.context_encoder = context_encoder
        self.context_encoder_target = context_encoder_target
        self.policy = policy #TanhGaussianpolicy
        self.use_next_state = kwargs['use_next_state']
        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        #self.use_ib = False
        self.sparse_rewards = kwargs['sparse_rewards']
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()
    def clear_z_sample(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)

        # sample a new z from the prior
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(mu), torch.unbind(var))]
            z = [d.rsample() for d in posteriors]
            z = torch.stack(z)
        else:
            z = mu
        return z
    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        #e = ptu.from_numpy(e[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if self.use_next_state:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)
    def explore_reward(self, z_prev, context_):

        z_pos = context_
        #print("prev:", z_prev)
        logits = self.compute_logits(z_prev, z_pos)
        #FIXME
        labels = torch.arange(logits.shape[0]).long().to(logits.device)
        ###TODO
        #logits = ptu.get_numpy(logits)
        #labels = ptu.get_numpy(labels)
        #loss = self.cross_entropy_loss_oneout(logits, labels)
        ###
        loss = ptu.get_numpy(self.cross_entropy_loss(logits, labels))
        #####prev = ptu.get_numpy(logits[0, labels[0]])
        z_new = self.encode(self.context)
        #print("new:", z_new)
        logits_ = self.compute_logits(z_new, z_pos)
        labels_ = torch.arange(logits_.shape[0]).long().to(logits_.device)
        loss_ = ptu.get_numpy(self.cross_entropy_loss(logits_, labels_))
        #####new = ptu.get_numpy(logits_[0, labels_[0]])
        #####reward = new - prev

        reward = loss - loss_
        return reward
    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context, ema=False):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        if ema==True:
            with torch.no_grad():
                params = self.context_encoder_target(context)
        else:
            params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:#information bottleneck
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
            #print(self.z_means.size())
        self.sample_z()
    def infer_posterior_(self, context, ema=True):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        with torch.no_grad():
            params = self.context_encoder_target(context)

        params = params.view(context.size(0), -1, self.context_encoder_target.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:#information bottleneck
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in
                          zip(torch.unbind(z_means), torch.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            task_z = torch.stack(z)
        # sum rather than product of gaussians structure
        else:
            z_means = torch.mean(params, dim=1)
            task_z = z_means
        return task_z

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means


    def get_action(self, obs, deterministic=False): #used when sample action conditioned on z
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z

        obs = ptu.from_numpy(obs[None])

        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def encode(self, context, ema=False):
        if ema==False:

            self.infer_posterior(context)
            self.sample_z()

            task_z = self.z
        else:
            task_z = self.infer_posterior_(context, ema=True)

        #task_z = [z.repeat(b, 1) for z in task_z] #dim: t * [b,-1]?
        #task_z = torch.cat(task_z, dim=0)#[t*b, -1]

        return task_z
    def forward(self, obs, context): #used during training given batch of experience (obs) output actions?
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z] #dim: t * [b,-1]?
        task_z = torch.cat(task_z, dim=0)#[t*b, -1]

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1) #calculate action needs stop gradient[t*b, -1]
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder, self.policy]
    def cross_entropy_loss_oneout(self, logits, labels, place=0):


        numerator = torch.exp(logits[0, labels[0]])
        

        denominator = torch.sum(torch.exp(logits[0, 1:]))

        loss = -torch.log(numerator / denominator)


        return loss

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class ExpPEARLAgent(nn.Module):#context encoder -> action output (during training and sampling)

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 context_encoder_target,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        #self.W = nn.Parameter(torch.rand(latent_dim, latent_dim))
        self.context_encoder = context_encoder
        self.context_encoder_target = context_encoder_target
        self.policy = policy #TanhGaussianpolicy
        self.use_next_state = kwargs['use_next_state']
        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        #self.use_ib = False
        self.sparse_rewards = kwargs['sparse_rewards']
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.task_z = None
        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        #e = ptu.from_numpy(e[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if self.use_next_state:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context, ema=False):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        if ema==True:
            with torch.no_grad():
                params = self.context_encoder_target(context)
        else:
            params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:#information bottleneck
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
            #print(self.z_means.size())
        self.sample_z()


    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means
    def copy_z(self, z):
        self.task_z = z

    def get_action(self, obs, deterministic=False): #used when sample action conditioned on z
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.task_z

        obs = ptu.from_numpy(obs[None])

        in_ = torch.cat([obs, z], dim=1)
        #in_ = obs
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)


    def forward(self, obs, task_z): #used during training given batch of experience (obs) output actions?
        ''' given context, get statistics under the current policy of a set of observations '''
        '''self.infer_posterior(context)
        self.sample_z()

        task_z = self.z'''
        self.z = task_z
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)


        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1) #calculate action needs stop gradient[t*b, -1]
        #in_ = obs
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)

        return policy_outputs

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder, self.policy]