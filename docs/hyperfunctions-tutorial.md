# A Basic Manual on Hyperfunctions and Cohomology

A **hyperfunction** is a generalization of the concept of a function, going even beyond distributions like the Dirac delta. The core idea, developed by Mikio Sato, is to think of a function on the real line, $f(x)$, as the "jump" or "boundary value difference" of a holomorphic function defined in the complex plane off of the real axis.

---

## The Intuitive Idea: Boundary Values

Imagine a function $F(z)$ that is holomorphic (complex differentiable) everywhere in a neighborhood of the real axis, *except* possibly on the real axis itself. We can approach a point $x$ on the real axis from the upper half-plane ($\mathbb{C}_+$) or the lower half-plane ($\mathbb{C}_-$).

The hyperfunction $f(x)$ associated with $F(z)$ is informally defined as the difference between these boundary values:

$$f(x) := F(x + i0) - F(x - i0) = \lim_{\epsilon \to 0^+} \left( F(x + i\epsilon) - F(x - i\epsilon) \right)$$

This "jump" across the real axis is the hyperfunction. We denote the hyperfunction represented by $F(z)$ as $[F]$.



---

## A More Formal Definition

This intuitive idea is formalized using the language of sheaf cohomology.

Let $U$ be an open subset of the real line $\mathbb{R}$. Let $V$ be an open neighborhood of $U$ in the complex plane $\mathbb{C}$. The space of **hyperfunctions** on $U$, denoted $\mathcal{B}(U)$, is defined as the quotient space:

$$\mathcal{B}(U) = \frac{\mathcal{O}(V \setminus U)}{\mathcal{O}(V)}$$

Here:
* $\mathcal{O}(A)$ is the space of holomorphic functions on the open set $A \subset \mathbb{C}$.
* $\mathcal{O}(V \setminus U)$ is the space of functions that are holomorphic on $V$ except for a singularity on $U$.
* $\mathcal{O}(V)$ is the space of functions that are holomorphic on all of $V$.

The quotient means that we consider two holomorphic functions $F_1, F_2 \in \mathcal{O}(V \setminus U)$ to define the **same** hyperfunction if their difference, $F_1 - F_2$, can be extended to a holomorphic function on all of $V$. In other words, $F_1 - F_2$ has no "jump" across $U$, so $[F_1] = [F_2]$.

### Connection to Cohomology: Unpacking $H^1_U(\mathbb{C}, \mathcal{O})$

This definition is a concrete realization of the first local cohomology group of $\mathbb{C}$ with supports in $U$ and coefficients in the sheaf of holomorphic functions, $\mathcal{O}$:

$$\mathcal{B}(U) = H^1_U(\mathbb{C}, \mathcal{O})$$

This notation measures the obstruction to extending a holomorphic function defined on $\mathbb{C} \setminus U$ to one defined on all of $\mathbb{C}$. That obstruction *is* the hyperfunction living on $U$. The other cohomology groups, $H^p_U(\mathbb{C}, \mathcal{O})$ for $p \neq 1$, are all zero, which makes this definition very natural.

The elements of this group **are** the hyperfunctions. Each hyperfunction is an **equivalence class** $[F]$ of holomorphic functions, where $[F_1] = [F_2]$ if and only if $F_1 - F_2$ is holomorphic and has no singularity on $U$.

---

## Key Examples

### The Dirac Delta Function: $\delta(x)$

The delta function is represented by $F(z) = -\frac{1}{2\pi i z}$. The hyperfunction $\delta(x)$ is the equivalence class $[-\frac{1}{2\pi i z}]$. Functions like $-\frac{1}{2\pi i z} + \sin(z)$ represent the same hyperfunction. Its action on a test function $\phi(x)$ is given by integrating the jump:
$$\langle \delta, \phi \rangle = \int (F(x+i\epsilon) - F(x-i\epsilon))\phi(x) dx \to \phi(0)$$

### The Heaviside Step Function: $H(x)$

The Heaviside function ($0$ for $x<0$, $1$ for $x>0$) is represented by $F(z) = -\frac{1}{2\pi i} \log(-z)$. The jump in the imaginary part of the logarithm across the positive real axis creates the value of 1.

### A Non-Distribution Hyperfunction

A hyperfunction like the one represented by $F(z) = e^{1/z}$ is supported only at the origin. However, its expansion $e^{1/z} = \sum \frac{1}{n! z^n}$ contains terms of arbitrarily high order, corresponding to an infinite series of derivatives of the delta function. A distribution must be of finite order locally, so this object is a hyperfunction but not a distribution.

---

## Operations on Hyperfunctions

Calculus on hyperfunctions is defined via their representing functions.

### Differentiation

The derivative of a hyperfunction $[F(z)]$ is simply $[F'(z)]$. For example, differentiating the representative for $H(x)$ gives the representative for $\delta(x)$:
$$ \frac{d}{dx} \left[ -\frac{1}{2\pi i} \log(-z) \right] = \left[ -\frac{1}{2\pi i z} \right] $$

### Multiplication by a Real Analytic Function

A hyperfunction $[F(z)]$ can be multiplied by a real analytic function $\phi(x)$ (which can be extended to a holomorphic $\phi(z)$):
$$ \phi(x) \cdot [F(z)] := [\phi(z)F(z)] $$

### Support of a Hyperfunction

The **support** of a hyperfunction $[F(z)]$ is the smallest closed set on the real line where the "jump" actually occurs. Outside the support, $F(z)$ is holomorphic. For $\delta(x)$, the support is $\{0\}$.
