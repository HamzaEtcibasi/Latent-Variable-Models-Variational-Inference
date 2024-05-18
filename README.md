# Latent Variable Models & Variational Inference
Topic Summary for CENG796 by Enes Şanlı &amp; Hamza Etcibaşı

## 1. Introduction and Motivation
- **What are Latent Variable Models (LVMs)?**
Latent Variable Models (LVMs) are statistical models that include variables that are not directly observed but are inferred from other variables that are observed (measured). These unobserved variables are termed "latent variables." LVMs are used to model complex phenomena where the observed data is believed to be generated from underlying factors that are not directly measurable.
![images/img1.png](img1.png "Fig 1. Latent Variables")
Like in Fig 1. suppose we want to generate an image of a dog. We know that dogs have certain features, such as color, breed, and size. However, can we limit these features? Or can we identify every feature for each image? The answer to this question is, of course, no. A single image can have an infinite number of latent features, and it is impossible for us to identify all of them accurately. However, if we can learn the most important of these features, we can use them to generate images much more easily. This is because estimating probability distributions based on an image's features is much easier than estimating from a complete probability distribution. This logic is the motivation behind Latent Variable Models.

- **Importance and Applications in Machine Learning and Statistics**
- **Motivation Behind Using Latent Variable Models**

## 2. Types of Latent Variables
#### 2.1 Mixture of Gaussians: A Shallow Latent Variable Model

The Mixture of Gaussians (MoG) model is a probabilistic model that assumes data points are generated from a mixture of several Gaussian distributions. Each Gaussian distribution represents a cluster or component within the overall data distribution. This model is often used for clustering and density estimation.

In a Mixture of Gaussians model, we have a latent variable \(z\) that determines which Gaussian component a data point belongs to. The generative process can be represented as a simple Bayesian network: z → x 

Here:
- \(z\) is the latent variable.
- \(x\) is the observed data.

![mog.png](mog.png "Fig 2. Mixture of Gaussians")
#### Generative Process

1. **Select a Gaussian component:**
   - Sample \( z \) from a categorical distribution with \( K \) components.
   <p align="center">
        $(z \sim \text{Categorical}(1, \ldots, K))$
   </p>

2. **Generate a data point:**
   - Given \( z = k \), sample \( x \) from the corresponding Gaussian distribution.
   <p align="center">
        $(p(x \mid z = k) = \mathcal{N}(x \mid \mu_k, \Sigma_k))$
   </p>

BU KISMA DEVAM EDECEĞİM YARIN


- **Use Cases and Advantages of MoG in Modeling Data Distributions**

#### 2.2 Variational Autoencoders (VAEs)
- **Explanation of VAEs as a Generative Model**
- **Difference Between Deterministic and Stochastic Latent Representations**

## 3. Inference and Marginal Probability
#### 3.1 Marginal Likelihood
- **Importance of Marginal Likelihood in Variational Probabilistic Modeling**

#### 3.2 Sampling Techniques
- **Overview of Monte Carlo Methods for Estimating Marginal Probabilities:**
  - Naive Monte Carlo
  - Importance Sampling

#### 3.3 Evidence Lower Bound (ELBO)
- **Introduction to ELBO as an Objective Function in VAEs**
- **ELBO’s Role in Variational Inference and Model Training**

## 4. Learning Latent Variable Models
#### 4.1 Stochastic Variational Inference (SVI)
- **Explanation of SVI and Its Role in Approximate Inference**

#### 4.2 Reparameterization Trick
- **Detailed Explanation of Reparameterization for Continuous Latent Variables**

#### 4.3 Amortized Inference
- **Introduction to Amortization Techniques for Efficient Inference in VAEs**

## 5. Autoencoder Perspective
#### 5.1 Comparing VAE with Traditional Autoencoders
- **Contrast Between VAEs and Standard Autoencoders in Representation Learning**
- **Advantages of VAEs in Generating Diverse and Meaningful Samples**

## 6. Conclusion
- **Summary of Key Concepts Covered in the Discussion of Latent Variable Models**
- **Pros and Cons of Latent Variable Models**
- **Future Directions and Challenges in the Field of Latent Variable Modeling**
