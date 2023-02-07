from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

import uncertainty
import mlp
import utils

def dense(input, A, b, ctx):
    if(uncertainty.is_const(input)):
        out = jnp.dot(input[0], A)
        if b is not None:
            out += b
        return  out, None, None, None

    mu, vecs, sigma, err = input

    def dot(x, with_abs=False):
        myA = jnp.abs(A) if with_abs else A 
        return jnp.dot(x, myA)
 
    mu = dot(mu)
    vecs = jax.vmap(dot)(vecs)
    sigma = jax.vmap(dot)(sigma)
    err = dot(err, with_abs=True)

    if b is not None:
        mu += b

    return mu, vecs, sigma, err
mlp.apply_func['uncertainty']['dense'] = dense

def relu(input, ctx):
    mu, vecs, sigma, err = input

    if uncertainty.is_const(input):
        return jax.nn.relu(mu), vecs, sigma, err

    mu, sigma = uncertainty.radius(input)

    # Compute the linearized approximation
    E = jnp.e ** (- mu * mu / 2 / sigma /sigma) / jnp.sqrt(2 * jnp.pi)
    C = jax.scipy.special.erf(-mu / jnp.sqrt(2) / sigma)
    beta = E * sigma
    alpha = (1-C) / 2

    # A = mu /2 * (1-C) + beta 
    # An = mu /2 * (1+C) - beta #int(-\inf, 0) px*x

    # B = (mu * mu + sigma * sigma) /2 * (1-C) + mu * beta #int(0,\inf) px*x
    # Bn = (mu * mu + sigma * sigma) /2 * (1+C) - mu * beta #int(-\inf, 0) px*x^2

    # target function
    # delta = alpha ** 2 * Bn + 2 * alpha * beta * An + \
    #     (1-alpha) ** 2 * B - 2* (1-alpha) * beta * A + \
    #     beta * beta
    delta = (1-C*C) * (mu * mu + sigma * sigma) / 4 + C*mu*beta - beta * beta # simplify delta
    
    delta = jnp.where(delta<0, 0, delta)
    delta = jnp.sqrt(delta)

    output = uncertainty.apply_linear_approx(ctx, input, alpha, beta, delta)
    return output
mlp.apply_func['uncertainty']['relu'] = relu

def elu(input, ctx):
    # Confusingly, elu has a parameter typically called 'alpha', and we also use 'alpha' for our linearizaiton notation. Here we simply ignore and do not support elu's alpha.
    mu, vecs, sigma, err = input

    if uncertainty.is_const(input):
        return jax.nn.elu(mu), vecs, sigma, err

    mu, sigma = uncertainty.radius(input)
    e = jnp.e

    # Compute the linearized approximation
    C = jax.scipy.special.erf(-mu / jnp.sqrt(2) / sigma)
    E = e ** (- mu * mu / 2 / sigma /sigma) / jnp.sqrt(2 * jnp.pi)

    G = jax.scipy.special.erf((-2*sigma * sigma - mu)/ jnp.sqrt(2)/sigma)
    H = e ** (2 * mu + 2 * sigma * sigma)
    J = H * (G+1)
    J = jnp.where(jnp.isnan(J),0,J) # J should be 0 when H is too large, (L'Hôpital's rule)

    F = e ** (mu + sigma * sigma / 2)
    D = jax.scipy.special.erf((-sigma * sigma - mu)/ jnp.sqrt(2)/sigma)
    I = (1+D) * F
    I = jnp.where(jnp.isnan(I),0,I) # J should be 0 when F is too large, (L'Hôpital's rule)

    # A = I / 2 + (1-C) * mu / 2 + sigma * E - (C + 1) / 2
    # B = I * (mu + sigma * sigma) / 2 + (mu * mu + sigma * sigma) * (1-C) / 2 + mu * sigma * E - mu /2 * (1+C)
    # alpha = (B- mu * A ) / sigma / sigma
    # beta = ((mu * mu + sigma * sigma) * A - mu * B) / sigma /sigma

    alpha = I / 2 + (1-C)/2
    beta = (1-mu)*I/2 -(1+C) / 2 + sigma * E
    # delta1 =  I * (-1-beta-alpha*(mu + sigma*sigma)) \
    #     + J/2 \
    #     + (alpha * alpha *sigma * sigma+(mu*alpha+1+beta)*(mu*alpha+1+beta)) * (C+1) /2 \
    #     - (alpha * mu + 2 * beta)* E * sigma * alpha
    # delta2 = ((alpha-1)**2*sigma**2+((alpha-1)*mu+beta)**2)/2 * (1-C) \
    #     + (alpha-1)*sigma*((alpha-1)*mu+2*beta) * E

    # delta1 = (-1/2 - (1+sigma ** 2 ) * I / 2 -sigma * E + C*(sigma * sigma + mu + 1) / 2-(mu + sigma * sigma)/2) * I \
    #     + J/2 \
    #     + ((I/2+(1-C)/2)**2*sigma**2 + (I/2 + (mu + 1)*(1-C)/2+sigma*E)**2) * (C+1) /2\
    #     - (mu*(1-C)/2+(1-mu/2)*I-1-C+2*sigma*E)*(I/2+1/2-C/2)*E*sigma

    # delta = delta1 + delta2
    # expended using alpha an beta
    # delta = alpha**2*mu**2 + alpha**2*sigma**2 + alpha*C*mu**2 + alpha*C*sigma**2 - 2*alpha*E*mu*sigma - alpha*I*sigma**2 + \
    #     2*alpha*beta*mu + alpha*C*mu - alpha*I*mu - alpha*mu**2 - alpha*sigma**2 + beta*C*mu - 2*beta*E*sigma - C*mu**2/2 - \
    #     C*sigma**2/2 + E*mu*sigma + alpha*mu + beta**2 + beta*C - beta*I - beta*mu + G*H/2 + mu**2/2 + sigma**2/2 + beta + C/2 + H/2 - I + 1/2

    # expended to mu and sigma
    delta = -C*C*mu*mu/4 - C*C*sigma*sigma/4 + C*E*mu*sigma +C*I*sigma*sigma/2 - E*E*sigma*sigma - I*I*sigma*sigma/4 - C*C*mu/2 + \
        C*E*sigma + C*I*mu/2 - E*I*sigma - I*sigma*sigma/2 - C*C/4 + C*I/2 + \
        E*sigma + J/2 - I*I/4 - I*mu/2 + mu*mu/4 + sigma*sigma/4 - I/2 + mu/2 + 1/4

    delta = jnp.where(delta<0, 0, delta)
    delta = jnp.sqrt(delta)
    output = uncertainty.apply_linear_approx(ctx, input, alpha, beta, delta)
    return output
mlp.apply_func['uncertainty']['elu'] = elu

def sin(input, ctx):
    mu, vecs, sigma, err = input

    if uncertainty.is_const(input):
        return jnp.sin(mu), vecs, sigma, err

    mu, sigma = uncertainty.radius(input)

    # mu = jnp.full_like(mu, 0.3, dtype=jnp.float64)
    # sigma = jnp.full_like(mu, 0.9999, dtype=jnp.float64)

    e = jnp.e
    cos = jnp.cos(mu)
    sin = jnp.sin(mu)

    # Compute the linearized approximation
    C = e ** (- sigma * sigma / 2)
    # A = sin*C
    # B = (mu*sin+sigma*sigma*cos)*C

    alpha = cos * C
    beta = (sin - mu * cos) * C

    # delta = alpha*alpha*(sigma*sigma+mu*mu)+beta*beta+2*alpha*beta*mu-2*alpha*B-2*beta*A +\
    #     (1-jnp.cos(2*mu)*C*C*C*C)/2
    delta = (-sigma*sigma*cos*cos-sin*sin) * C*C + (1-jnp.cos(2*mu)*C*C*C*C)/2
    # print(mu[:5])
    # print(sigma[:5])
    # print(delta[:5])
    # exit()

    # floating point inaccuracy
    delta = jnp.where(delta<0, 0, delta)
    delta = jnp.sqrt(delta)
    output = uncertainty.apply_linear_approx(ctx, input, alpha, beta, delta)
    return output
mlp.apply_func['uncertainty']['sin'] = sin

def pow2_frequency_encode(input, ctx, coefs, shift=None):
    mu, vecs, sigma, err = input

    # TODO debug
    if len(mu.shape) > 1:
        raise ValueError("big base")

    # expand the length-d inputs to a lenght-d*c vector
    def s(with_shift, x): 
        out = (x[:,None] * coefs[None,:])
        if with_shift and shift is not None:
            out += shift
        return out.flatten()

    mu = s(True, mu)
    if uncertainty.is_const(input):
        return mu, None, None, None

    vecs = jax.vmap(partial(s, False))(vecs)
    sigma = jax.vmap(partial(s, False))(sigma)
    err = s(False, err)
    
    return mu, vecs, sigma, err
mlp.apply_func['uncertainty']['pow2_frequency_encode'] = pow2_frequency_encode

def squeeze_last(input, ctx):
    mu, vecs, sigma, err = input
    s = lambda x : jnp.squeeze(x, axis=0)
    mu = s(mu)
    if uncertainty.is_const(input):
        return mu, None, None, None
    vecs = jax.vmap(s)(vecs)
    sigma = jax.vmap(s)(sigma)
    err = s(err)
    return mu, vecs, sigma, err
mlp.apply_func['uncertainty']['squeeze_last'] = squeeze_last

def spatial_transformation(input, R, t, ctx):
    # if the shape transforms by R,t, input points need the opposite transform
    R_inv = jnp.linalg.inv(R)
    t_inv = jnp.dot(R_inv, -t)
    return dense(input, A=R_inv, b=t_inv, ctx=ctx)
mlp.apply_func['uncertainty']['spatial_transformation'] = spatial_transformation
