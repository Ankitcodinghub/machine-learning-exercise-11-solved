# machine-learning-exercise-11-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning Exercise 11 Solved](https://www.ankitcodinghub.com/product/machine-learning-labs-solved-7/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;110207&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning Exercise 11 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
&nbsp;

Let f : RD → R. Recall that the gradient of f is a (column) vector of length D whose d-th component is the derivative of f(x) with respect to . The Hessian is the D × D matrix whose entry (i,j) is the second derivative of f(x) with respect to xi and .

Let f : RD → R be the function f(x) = x⊤Ax + b⊤x + c, where A is a (possibly asymmetric) D × D matrix, b is a vector of length D and c is a constant.

1. Determine the gradient of f, ∇f(x).

2. Determine the Hessian of f, ∇2f(x).

1.2 Maximum Likelihood Principle

Assume we are given i.i.d. samples X1,··· ,XN ∈ R drawn from a Gaussian distribution with mean µ and variance σ2. We do not know the two parameters (µ, σ) and want to estimate them from the data using the maximum likelihood principle.

1. Write down the likelihood for this data, i.e., the joint distribution Pµ,σ2(X1,··· ,XN), where the subscripts µ and σ2 remind us that this distribution depends on these two parameters.

2. Use the maximum likelihood principle to estimate the two parameters µ and σ2. More precisely, find the values µˆ(X1,··· ,XN) and σˆ2(X1,··· ,XN) which maximize the likelihood that you computed in the previous question. (Hint: taking the logarithm of the likelihood leads to much simpler computations).

3. Compute E[µˆ]. Is this equal to the true parameter µ?

4. Compute E[σˆ2]. Is this equal to the true parameter σ2?

2 Implementing K-Means

Goals. The goal of this exercise is to

• Implement and visualize K-means clustering using the faithful dataset.

• Visualize the behavior with respect to the number of clusters K.

• Implement data compression using K-means.

Setup, data and sample code. Obtain the folder labs/ex11 of the course github repository

github.com/epfml/ML course

We will use the dataset faithful.csv in this exercise, and we have provided sample code templates that already contain useful snippets of code required for this exercise.

We will reproduce Figure 9.1 of Bishop’s book.

Exercise 2a):

Let’s first implement K-means algorithm using the faithful dataset.

• Fill-in the code to initialize the cluster centers.

• Write the function kmeansUpdate to update the assignments z, the means µ, and the distance of data points to the means. Your code should work for any number of clusters K (not just K = 2).

• Write code to test for convergence.

• Visualize the output. You should get figures similar to Figure 1.

(a) Iteration 0 (b) Iteration 1

(c) Iteration 2 (d) Iteration 3

Figure 1: K-means for faithful data.

Exercise 2b):

Now, play with the initial conditions and the number of clusters to understand the behavior of K-means. • Change the initial conditions and observe the change in convergence. The algorithm must converge for all possible initial conditions, otherwise there is a problem in your implementation.

• Try different values for K. Also try different values of initial condition. Look at the cost function value as K increases.

• BONUS: What is a good value for K? How will you choose it?

3 Data Compression using K-Means

We will implement data compression using K-means, similar to the examples shown in the class.

2

Exercise 3:

Write data compression for mandrill.png.

Your output should look like Figure 2.

Run K-means with random initializations and observe the convergence. Plot the reconstructed image by setting each pixel’s value to the mean value of its cluster. Play with the number of clusters and compare the compression you get in your resulting image.

Figure 2: Image quantization / compression using K-means.

3
