# Phase 7 — De-slop demo (Gemma 2 2B-it, feature ablation)

**Features ablated:** [3223]
**Pairs:** 36 (12 D2 prompts × 3 seeds)
**Sampling:** temperature=0.8, top_p=0.95, max_new_tokens=100

## Construction rate (any-core C1∪C2∪C3)

- baseline (no intervention): **5.56%** (2/36 generations contain the construction)
- ablated (feature → 0):     **5.56%** (2/36)
- absolute drop:             **+0.00%**
- relative drop:             **+0.00%**

## Meaning preservation (MiniLM cosine of baseline vs ablated)

- mean cosine = 0.954, median = 0.986
- 10th pct = 0.871, 90th pct = 1.000

Meaning is largely preserved (mean cosine ≥ 0.75) — the ablation is a scalpel, not a sledgehammer.

## Examples (first 6 pairs)

### `Explain why a city might invest in public transit infrastructure.` (seed 0)

**Baseline** ·:  
> **Answer:**

There are many reasons why a city might invest in public transit infrastructure. Here are some of the most important:

* **Economic Growth and Job Creation:** Public transportation is a vital economic driver. It increases accessibility to jobs, strengthens local economies, and boosts overall productivity. 
* **Reduced Congestion and Traffic Delays:** Investing in public transportation

**Ablated** ·:  
> **Answer:**

There are many reasons why a city might invest in public transit infrastructure. Here are some of the most important:

* **Economic Growth and Job Creation:** Public transportation is a vital economic driver. It increases accessibility to jobs, strengthens local economies by attracting workers and businesses, and encourages investment in areas with better transportation infrastructure

---

### `Explain why a city might invest in public transit infrastructure.` (seed 1)

**Baseline** ·:  
> **Public Transit Investment: Why Invest?**

Cities invest in public transportation for a variety of reasons, and these reasons often intertwine. Here are some of the most common reasons:

* **Reduced Traffic Congestion:** One of the most significant benefits of public transportation is its ability to alleviate traffic congestion. Investing in public transportation like trains, buses, and light rai

**Ablated** ·:  
> **Public Transit Investment: Why Invest?**

Cities invest in public transportation for a variety of reasons, and these reasons often intertwine. Here are some of the most common reasons:

* **Reduced Traffic Congestion:** One of the most significant benefits of public transportation is its ability to alleviate traffic congestion. Investing in public transportation like trains, buses, and light rai

---

### `Explain why a city might invest in public transit infrastructure.` (seed 2)

**Baseline** ·:  
> **Why cities invest in public transportation is because of the following reasons:**

**Economic Factors:**
* **Reduced Traffic Congestion:** Public transportation reduces the number of cars on the road, alleviating traffic congestion, leading to increased efficiency and reduced travel time. This translates to reduced fuel consumption and lower emissions.
* **Increased Productivity:** Public transp

**Ablated** ·:  
> **Why cities invest in public transportation is because of the following reasons:**

**Economic Factors:**
* **Reduced Traffic Congestion:** Public transportation reduces the number of cars on the road, alleviating traffic congestion, leading to increased efficiency and reduced travel time. This translates to reduced fuel consumption and lower emissions.
* **Increased Productivity:** Public transp

---

### `Describe what makes a good neighbourhood library.` (seed 0)

**Baseline** ·:  
> A good neighborhood library isn't just a place to check out books. It's a community center that celebrates knowledge, creativity, and community. 
 
Here are some key aspects that make a good neighborhood library stand out:
 
**Accessibility and Community Engagement:**
* **High-quality library services:** The key is providing a diverse selection of books, technology, and resources for all ages. Thi

**Ablated** ·:  
> A good neighborhood library isn't just a place to check out books. It's a community center that celebrates knowledge, creativity, and community. 
 
Here are some key aspects that make a good neighborhood library stand out:
 
**Accessibility and Community Engagement:**
* **High-quality library services:** The key is providing a diverse selection of books, technology, and resources for all ages. Thi

---

### `Describe what makes a good neighbourhood library.` (seed 1)

**Baseline** ·:  
> 

**Ablated** ·:  
> 

---

### `Describe what makes a good neighbourhood library.` (seed 2)

**Baseline** ·:  
> A good neighborhood library is more than just a place to borrow books. It's a hub for the community, offering a wide range of services and activities. Here are some of the key features that make a good library:
 
 **Accessibility and Location:**
  *  **Convenient Location:** The library should be easily accessible by car, bus, and train, and easily recognizable from the street. 
  
  **Collections

**Ablated** ·:  
> A good neighborhood library is more than just a place to borrow books. It's a hub for the community, offering a wide range of services and activities. Here are some of the key features that make a good library:
 
 **Accessibility and Location:**
  *  **Convenient Location:** The library should be easily accessible to the community.
  
**Services and Programming:**
  * **Wide Range of Materials:** 

---
