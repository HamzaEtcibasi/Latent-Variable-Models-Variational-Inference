# Latent Variable Models & Variational Inference
Topic Summary for CENG796 by Enes Şanlı &amp; Hamza Etcibaşı

## 1. Introduction and Motivation
- **What are Latent Variable Models (LVMs)?**
Latent Variable Models (LVMs) are statistical models that include variables that are not directly observed but are inferred from other variables that are observed (measured). These unobserved variables are termed "latent variables." LVMs are used to model complex phenomena where the observed data is believed to be generated from underlying factors that are not directly measurable.
![img1.png](images/img1.png "Fig 1. Latent Variables")
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

<p align="center">
  <img src="images/mog.png" alt="Fig 2. Mixture of Gaussians" title="Fig 2. Mixture of Gaussians" width="37%">
</p>

#### Generative Process

1. **Select a Gaussian component:**
   - Sample **z** from a categorical distribution with **k** components.
   <p align="center">
        $z \sim \text{Categorical}(1, \ldots, k)$
   </p>

2. **Generate a data point:**
   - Given **z = k**, sample **x** from the corresponding Gaussian distribution.
   <p align="center">
        $p(x \mid z = k) = \mathcal{N}(x \mid \mu_k, \Sigma_k)$
   </p>

When training a Mixture of Gaussians model, we start with like a single Gaussian fit. As training progresses, assuming the distribution isn't too complex and we have a good estimate for the number of clusters, the model performs effective clustering. Similar points end up with similar z estimates, resulting in soft clustering. This approach is also useful in unsupervised learning. The approach is illustrated in Fig 3. below:
<p align="center">
  <img src="images/mog2.png" alt="Fig 3. Mixture of 3 Gaussians" title="Fig 3. Mixture of 3 Gaussians" width="70%">
</p>

By following these steps, the Mixture of Gaussians model generates data points that can represent complex, multimodal distributions through the combination of multiple Gaussian components.

- **Use Cases and Advantages of MoG in Modeling Data Distributions**

The Mixture of Gaussians (MoG) model is a versatile and powerful tool in various data analysis tasks. Here are some key use cases and advantages of using MoG:

#### **Use Cases**

1. **Clustering:** MoG is commonly used for clustering data into groups. Each Gaussian component can be seen as representing a cluster, allowing for effective identification of subpopulations within the data.
   
2. **Density Estimation:** It is useful for estimating the probability density function of a dataset, especially when the data distribution is complex and multimodal (having multiple peaks).

3. **Anomaly Detection:** By modeling the normal data distribution with MoG, points that fall in low-probability regions can be identified as anomalies or outliers.

4. **Image Segmentation:** In image processing, MoG can be used to segment images by clustering pixels based on color or intensity values.

5. **Speech and Signal Processing:** MoG is applied in modeling different sound patterns or signals, aiding in tasks like speaker identification or noise reduction.

#### **Advantages**

1. **Flexibility:** MoG can model a wide range of data distributions by combining multiple Gaussian components, making it suitable for complex datasets.

2. **Soft Clustering:** Unlike hard clustering methods, MoG provides soft clustering where each data point has a probability of belonging to each cluster, offering more nuanced insights.

3. **Scalability:** MoG can be scaled to handle large datasets and high-dimensional data, making it practical for real-world applications.

4. **Interpretability:** The parameters of the Gaussian components (mean and covariance) provide meaningful information about the data structure, aiding interpretability.

5. **Probabilistic Framework:** Being a probabilistic model, MoG allows for uncertainty quantification and principled statistical inference.

By leveraging these advantages, the Mixture of Gaussians model is a powerful tool for understanding and analyzing complex data distributions.

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
