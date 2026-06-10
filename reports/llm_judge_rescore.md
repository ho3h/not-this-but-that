# Blind LLM-judge re-score

Judge: `claude-opus-4-8` via Message Batches, blinded to condition and
pairing, randomized order, structured-output schema. Spans quoted for
spot-audit. Batch id: `msgbatch_01GQSGBAatyssRPS5ey6eJax`.

| eval | n pairs | judge baseline | judge ablated | rel drop | mid-p | agreement w/ union |
|---|---:|---:|---:|---:|---:|---:|
| q5b | 300 | 54 | 28 | 48.1% | 0.00046 | 87.8% |
| q5c | 306 | 37 | 34 | 8.1% | 0.5034 | 92.0% |
| q5d | 120 | 28 | 27 | 3.6% | 0.845 | 79.6% |

Disagreements with the union detector (judge vs regex):

- **q5b/ablated** judge=True regex=False span="Instead, breaking is the process that happens when you have a specific tool designed to remove something that's already in place.":  that you use to "break" things in the way you might use a hammer to hit a nail. Instead, breaking is the process that happens when you have a specific tool des…
- **q5b/baseline** judge=False regex=True span=None: . She's a tough woman, a woman who knows what she wants. She's strong and resilient, she's not going to let anyone break her down. She's a survivor. 
 
This is …
- **q5b/baseline** judge=False regex=True span=None: , but I think you're describing a common phenomenon. 
I'm not sure what you mean by "tool", but here are some thoughts.

**Common Phenomenology:**

* **Cognitiv…
- **q5b/ablated** judge=True regex=False span="about whether we should believe in god. \nIt's about whether there is evidence for the existence of God.":  about whether we should believe in god. 
It's about whether there is evidence for the existence of God.
You can't deny the evidence of your own eyes.
You can't…
- **q5b/baseline** judge=False regex=True span=None: , but they're not awake either. 
They don't know what's happening.
They just exist.
They're in the middle,
Between dreams and reality.
Between consciousness and…
- **q5b/baseline** judge=False regex=True span=None: , it's about the journey, right? 
 
I'm not sure if I'm talking about life or my career, but that sentiment rings true in both, I think.
 
I'm struggling with t…
- **q5b/baseline** judge=False regex=True span=None: , but I can see tears welling up in her eyes.  I'm not sure what to do.
 
 I'm not sure if this is a reflection of me or her, but I want to help her feel okay a…
- **q5b/ablated** judge=True regex=False span='they are actually fraudsters':  service representative, they are actually fraudsters. I was scammed by one of them.  
 
I am hoping someone will help me with information about reporting this …
- **q5b/ablated** judge=True regex=False span="Sun Tzu wasn't just a strategist.":  "The Art of War" Sun Tzu'
   t read the writings of famous thinkers, philosophers, and historians of his time. 
   
   Sun Tzu wasn't just a strategist. 
   
 …
- **q5b/ablated** judge=True regex=False span="It's not a question of whether it's a good idea. It's a question of how to make it happen.":  of whether it's a good idea. It's a question of how to make it happen.
 
 You're thinking about starting a business. Good for you! It's a challenging, exciting…
- **q5b/baseline** judge=True regex=False span='that defines a person, but the way they react to what they see in front of them':  that defines a person, but the way they react to what they see in front of them, in this case, the violence.
 
The way people react to violence can define who …
- **q5b/baseline** judge=True regex=False span=', but a command!': , but a command!

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
…
- **q5b/ablated** judge=True regex=False span="It's not about the fashion industry. This is hair for people who work hard and strive to make a difference.": -haired, or the most glamorous of hairstyles. It's not about the fashion industry. This is hair for people who work hard and strive to make a difference.

This …
- **q5b/baseline** judge=True regex=False span='is not just visiting': . She's a local.

This is a common phrase that implies someone is deeply connected to their community and is not just visiting. 
 
Let me know if you'd like mor…
- **q5b/baseline** judge=True regex=False span='weather that we should be worried about, but the snow':  weather that we should be worried about, but the snow.
 
 "Snowstorms can be a real challenge for our communities," said a local official. "It's important to b…
- **q5c/ablated** judge=True regex=False span="It's not just a": 

As a writer specializing in outdoor adventures and breathtaking natural wonders like the Northern Lights (Aurora Borealis) seeing it for the&#39;s the first t…
- **q5c/baseline** judge=True regex=False span='I was expecting a grand spectacle, a grandiose opera house filled with plush seats and fancy costumes, but what I experienced was much more profound': 

For me, the experience of attending my first opera was a whirlwind of emotions, sensations, and perspectives. I was expecting a grand spectacle, a grandiose o…
- **q5c/baseline** judge=True regex=False span='independent bookstores are more than just brick-and': 

## The Survival of the Human-Book Store in the Digital Age

Independent bookstores are often viewed as relics of the past, facing stiff competition from the c…
- **q5c/ablated** judge=True regex=False span='finding a mentor is a journey, not a destination': 
 
 A well-structured discussion of this topic will include the following aspects:
   * **Benefits of mentorship:** what does it offer to the mentee?
   
   * *…
- **q5c/ablated** judge=True regex=False span="The best coffee shops go beyond just serving coffee. They become a part of the community's fabric.": 
** 
The best coffee shops go beyond just serving coffee. They become a part of the community's fabric.  Baristas play a crucial role in building this community…
- **q5c/ablated** judge=True regex=False span='A public library is much more than a place to get a book. It is a community hub for learning and creativity.': 
 
A public library is much more than a place to get a book. It is a community hub for learning and creativity.  A library provides a safe space for teens to ex…
- **q5c/baseline** judge=True regex=False span='more than just a barista; they are a part of the routine for their customers': 
 
**Answers:**
 
A great coffee shop barista is more than just a barista; they are a part of the routine for their customers, providing more than just a cup of…
- **q5c/baseline** judge=True regex=False span="They deliver more than just a graduation speech; they are a testament to one's journey and a glimpse into their future.": 
 
 
 
A memorable graduation speech is a blend of personal insights, impactful narratives, and powerful delivery. They deliver more than just a graduation spee…
- **q5c/ablated** judge=True regex=False span='It goes beyond the mere feeling of accomplishment.': 
 
 The satisfaction of finishing a long novel is complex and multi-faceted.  It goes beyond the mere feeling of accomplishment.  It's a blend of emotional and …
- **q5c/ablated** judge=True regex=False span="Making your own bread is more than just a recipe. It's a journey of understanding the ingredients' impact on the final outcome.": 

Making your own bread is more than just a recipe. It's a journey of understanding the ingredients' impact on the final outcome. The knowledge of baking techni…
- **q5c/baseline** judge=True regex=False span='go beyond just serving coffee; they become part of the day': 
** 
The best coffee shops go beyond just serving coffee; they become part of the day.
Here are some ways a great coffee shop barista contributes to a regular's…
- **q5c/baseline** judge=True regex=False span='Public libraries are more than just storage spaces for books; they are vital hubs for young minds.': 
 
**Answer:**

Public libraries are more than just storage spaces for books; they are vital hubs for young minds. Here's how they shape a teenager's reading li…
- **q5c/baseline** judge=True regex=False span='much more than just places to grab a coffee. They are often hubs for community engagement, creative expression, and connection':  
 
The best coffee shops can be much more than just places to grab a coffee. They are often hubs for community engagement, creative expression, and connection.…
- **q5c/ablated** judge=True regex=False span='A great coffee shop barista is more than just a barista.': 
 
**Answers:**
 
A great coffee shop barista is more than just a barista. They are a part of the daily routine for many people. They provide a small touch of w…
- **q5c/baseline** judge=True regex=False span='Soundtracks are more than just background music in a film. They are integral to its storytelling': 

Soundtracks are more than just background music in a film. They are integral to its storytelling, often influencing how we experience the film's plot, charact…
- **q5d/ablated** judge=False regex=True span=None:  to find, but he's difficult to understand. He's not easy to talk to. He's not always prepared for the conversation.

What is this person?

**Answer:** A person…
- **q5d/baseline** judge=True regex=False span='about whether we should believe in god, but about the question of whether or not the secular world can embrace the godless future':  about whether we should believe in god, but about the question of whether or not the secular world can embrace the godless future or if we should embrace the r…
- **q5d/baseline** judge=True regex=False span=", it's the result of hard work and dedication.": , it's the result of hard work and dedication.

**I'm not sure what you mean by "luck." I'm not sure if you meant to say you're proud of your work, or you're si…
- **q5d/ablated** judge=True regex=False span='She writes with her heart and not with her words.':  or eloquent, but her voice echoes with a kind of quiet power that makes her stories captivating.  She writes with her heart and not with her words. 
 
This is …
- **q5d/baseline** judge=False regex=True span=None: , but I think you're describing a common phenomenon. 
I'm not sure what you mean by "tool", but here are some thoughts.

**Common Phenomenology:**

* **Cognitiv…
- **q5d/ablated** judge=True regex=False span='This is a statement of opinion, not a fact.':  about the world. 
 
This is a statement of opinion, not a fact.  Let's look at why:

* **Opinion vs. Fact:** A fact is a statement that can be proven true or f…
- **q5d/baseline** judge=False regex=True span=None: , but they're not awake either. They're somewhere in between, in a dream-like state, and maybe you can even see a little bit of the light from the outside world…
- **q5d/ablated** judge=True regex=False span='with the code itself, but rather with how the code is being used':  with the code itself, but rather with how the code is being used. 
  
   Let me know how I can help. 
   
   
   
   
   
   
   
   
   
   
…
- **q5d/baseline** judge=False regex=True span=None: , but I enjoyed the story. The narrative was engaging and well-written. The suspense kept me captivated and I couldn't put it down. 
 
I didn't necessarily unde…
- **q5d/ablated** judge=True regex=False span='about money. \n \n The argument is about the **meaning of the art** and the relationship between the artist and the public.':  about money. 
 
 The argument is about the **meaning of the art** and the relationship between the artist and the public. 
 
 The idea that art should be purel…
- **q5d/baseline** judge=True regex=False span=', but a paradigm.': , but a paradigm.

Let's take "machine learning" as an example.  A tool, like a hammer, is a specific object that performs a specific task. A specific task is s…
- **q5d/baseline** judge=True regex=False span="for the day to end, but instead, we're taking full advantage of the time we have now":  for the day to end, but instead, we're taking full advantage of the time we have now. This is the life I'm living, and I'm loving it.

This is what I believe:
…
- **q5d/ablated** judge=True regex=False span="it's the opposite": ; it's the opposite. The bottom line is that the tech industry should be focused on improving existing technologies rather than chasing after things that are no…
- **q5d/baseline** judge=False regex=True span=None: , but I'm not sure what else to do.

**I'm not sure what to do with my life.**

**It's not a deadline, but I'm not sure what else to do.**

I get it…
- **q5d/baseline** judge=True regex=False span='with the code itself, but rather with how the code is being used.':  with the code itself, but rather with how the code is being used. 
  
   Let me know how I can help. 
   
   
   
   
   
   
   
   
   
   
…
