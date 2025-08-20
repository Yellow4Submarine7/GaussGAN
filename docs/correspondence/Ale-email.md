Tiesunlong,
AMAZING!! :)))

Incredible that works only with 20 iterations! 

Looking forward to learning a lot this friday! 

Ale. 

On Mon, 24 Mar 2025 at 11:18, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:
Dear Dr. Ale.,

Hope you're doing well! I've been playing with our GaussGAN project and made some improvements after quite a bit of trial (since I'm not very familiar with GAN networks~). Regarding the issue of WGAN not generating nice Gaussians, I've spent some hours experimenting and finally improved it by adjusting the network structure, using LeakyReLU activation functions, and optimizing batch sizes and learning rates. Now the visualization shows generated data points reasonably distributed across the target distribution (not 100% perfect yet, and there's still room for improvement).

The quantum circuits and quantum shadows noise generators are working perfectly now too (they're so slow though😂, especially the quantum shadows~).

As for the killer function you mentioned, I've got it working. It now successfully kills the distribution on the negative x-axis. One thing I noticed is that with my current implementation the killed points don't completely disappear but instead move to the left portion of the target distribution on the positive x-axis - this is another area where we could make further improvements.

While there's still some room for improvement, I think we've accomplished most of what we set out to do. What do you think? You can check out the visualization results in the attachment.

Best,
Tiesunlong



-----原始邮件-----
发件人: "Alessandro Luongo" <alessandro@luongo.pro>
发送时间: 2025-03-19 15:06:42 (星期三)
收件人: "Tiesunlong Shen" <tensorshen@mail.ynu.edu.cn>
主题: Re: Thoughts on GaussGAN and Quantum Circuits Integration

Tiesunlong
you are absolutely correct on everything. 



>I've recently been studying your GaussGAN code, particularly the quantum circuit part. As I understand, our goal is to replace the traditional generator (neural network) in the GAN with a quantum circuit, allowing it to learn to generate the target distribution. The difference is that neural networks are parameterized by weights and biases, while quantum circuits are parameterized by rotation angles and other quantum gate parameters, but during training, both can have their parameters updated through gradient descent and backpropagation. Is my understanding correct?

Yes.

If the above holds true, then theoretically we can replace any neural network with a quantum circuit, right?

Yes. I don't believe it is an easy task, but most people do. There are billions of problems of expressivity and trainability (e.g. barren plateaus - https://arxiv.org/abs/2312.09121 if you are interested in the skepticism about this). 



Regarding the issues we're facing:
1.In the current implementation, KDE is used to estimate probability distributions from generated sample points, but KDE estimates may not be accurate enough when the sample size is small. Perhaps we could try k-Nearest Neighbors (k-NN), Wasserstein distance, or the variational lower bound (ELBO) for estimation?

indeed. I tried the first thing proposed by chatgpt :) However, this is only to "see" easily from mflow the "best" training, and is not entering the loss function. 

Another issue is that the current calculation is KL(P||Q), while in generative model evaluation, we typically care more about KL(Q||P), i.e., the KL divergence from the target distribution Q to the generated distribution P. These are different perspectives.

You are totally right and I didn't think about it! I am trying the KL as the loglikelihood is not a "good" metric for the validation as 1000 points overlapping with the centroid of the gaussians have high likelihood, but they are clearly not from the training distribution. 





2.For the requirement to "kill" one Gaussian distribution, I think we should train a "value network" to accomplish this.

Totally correct. There should already be a network in the code that at the moment is turned off. 



These are my initial thoughts on the project. I've spent some time understanding the concepts of quantum circuits and how they integrate with neural networks (especially in the QuantumShadowNoise class in nn.py). These are new areas for me, and I'll continue to delve deeper into the other components of the project.

Yeah Shadow Tomography is not easy. Just think about it as a way to have (for some circuits) exponentially many estimates of interesting values (observables) at a linear cost (where we cost quantum stuff in number of samples, as a sample requires a new circuit to be trained). 





I look forward to continuing this exploration and would greatly appreciate your feedback on these ideas! 

We can take some time next Friday to play with this! I hope I'll be able to generate "nice" gaussians. If you run the script for the plots, the gan is not generating "nice" gaussian that are overlapping "nicely" with the training set.



I hope everything at home is going well. 

What kind of visa are you getting for april? 

Ale. 

On Wed, 19 Mar 2025 at 06:31, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:
Dear Dr. Ale.,

I've recently been studying your GaussGAN code, particularly the quantum circuit part. As I understand, our goal is to replace the traditional generator (neural network) in the GAN with a quantum circuit, allowing it to learn to generate the target distribution. The difference is that neural networks are parameterized by weights and biases, while quantum circuits are parameterized by rotation angles and other quantum gate parameters, but during training, both can have their parameters updated through gradient descent and backpropagation. Is my understanding correct?

If the above holds true, then theoretically we can replace any neural network with a quantum circuit, right?

Regarding the issues we're facing:
1.In the current implementation, KDE is used to estimate probability distributions from generated sample points, but KDE estimates may not be accurate enough when the sample size is small. Perhaps we could try k-Nearest Neighbors (k-NN), Wasserstein distance, or the variational lower bound (ELBO) for estimation?
Another issue is that the current calculation is KL(P||Q), while in generative model evaluation, we typically care more about KL(Q||P), i.e., the KL divergence from the target distribution Q to the generated distribution P. These are different perspectives.

2.For the requirement to "kill" one Gaussian distribution, I think we should train a "value network" to accomplish this.

These are my initial thoughts on the project. I've spent some time understanding the concepts of quantum circuits and how they integrate with neural networks (especially in the QuantumShadowNoise class in nn.py). These are new areas for me, and I'll continue to delve deeper into the other components of the project.


I look forward to continuing this exploration and would greatly appreciate your feedback on these ideas!

Best regards,
Tiesunlong


-----原始邮件-----
发件人: "Alessandro Luongo" <alessandro@luongo.pro>
发送时间: 2025-03-12 05:08:55 (星期三)
收件人: "Tiesunlong Shen" <tensorshen@mail.ynu.edu.cn>
主题: Re: Introduction from Prof. Erik Cambria - Tiesunlong Shen (Tensorlong)

Hello Tiesunlong,
Not sure I will be in Singapore in early April. 

What about you start having a look at this? https://github.com/Scinawa/GaussGAN before playing with quantum things, I would love to be able to train the GAN. However, it's not so easy. Do you want to play with me and understand why this WGAN does not generate nice gaussians? 

I am not sure the KLDivergence metric is correct, perhaps there are better algorithms to compute the KL between samples and a distribution (and vice versa?) . 
Then, we want to enable a "value network" that "kills" one of the two gaussians (e.g. the one with negative values on the x axis). 

Then, we want to sample from different distributions (e.g. quantum circuits, quantum shadows, tomography..). 

We can have a chat with Amine in ~2 weeks and talk about this and your previous work, what do you think?  I am happy to also talk about it in the next days, because I will be playing with it as well. 

I am doing this for learning, as I never had a chance to play with real NN. :) 

Ciao,
Ale. 




On Tue, 11 Mar 2025 at 08:52, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:

Dear Dr. Ale.,


Thank you for your response. I'm happy to schedule a technical interview with your collaborator. Could you please provide some details about the interview format and any specific topics I should prepare for?


Regarding deadlines, I don't have any urgent time constraints. The timing actually works well for me - I'll be traveling to Singapore in early April, so if it's convenient for you, we could potentially arrange an in-person interview during my visit.


I'm looking forward to the opportunity to learn more about your team's work and discuss potential collaboration.


Best regards,
Tiesunlong

-----原始邮件-----
发件人: "Alessandro Luongo" <alessandro@luongo.pro>
发送时间: 2025-03-11 10:08:13 (星期二)
收件人: "Tiesunlong Shen" <tensorshen@mail.ynu.edu.cn>
主题: Re: Introduction from Prof. Erik Cambria - Tiesunlong Shen (Tensorlong)

Tiesunlong,
I hope you are well! 

We didn't even start a proper search at the moment.

Do you have particular deadlines? If not, propose we schedule a technical interview with my collaborator in 2 weeks? 

Best,
Ale. 





On Mon, 10 Mar 2025 at 13:43, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:
Dear Dr. Ale.,


Hope you're doing well! I want to follow up about the internship opportunity we discussed a couple of weeks ago.


I was wondering if there have been any updates on the decision process? I'm still very excited about the possibility of joining your team in Singapore and contributing to your work.


Thanks for your consideration!


Best regards,
Tiesunlong



-----原始邮件-----
发件人: "Alessandro Luongo" <alessandro@luongo.pro>
发送时间: 2025-02-24 14:36:18 (星期一)
收件人: "Tiesunlong Shen" <tensorshen@mail.ynu.edu.cn>
主题: Re: Introduction from Prof. Erik Cambria - Tiesunlong Shen (Tensorlong)

Hello Tiesunlong Shen,

I think the only option is an internship, because you cannot be a research assistant if you are employed by your university as a PhD student. 
Is this ok for you?

At the moment you are our favourite candidate, and we will make a decision in the next two weeks!  Looking forward to have you here in Singapore soon, anyway! 

Best,
Ale. 

On Mon, 24 Feb 2025 at 14:18, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:
Dear Dr. Ale,


I hope this email finds you well. I apologize for my delayed response. 


This morning, I discussed my plans for early graduation with my department and academic 

committee, and they have tentatively agreed to allow me to complete my PhD by December of 

this year. 


As a result, I am now seeking a postdoctoral position for early next year. I would be truly 

honored if I could initially join your team as a research assistant for six months. Your 

research direction and the innovative projects at your lab deeply resonate with my academic 

interests. If you find my work satisfactory after I obtain my PhD, I would be extremely 

eager to compete for a postdoctoral position with your group. I strongly believe that a 

continuous research collaboration would create more value and allow me to make meaningful 

contributions to your ongoing projects.


I am very excited about the possibility of working with you and your team. What do you think

of this proposal? I look forward to your response.


Best wishes,
Tiesunlong Shen




-----原始邮件-----
发件人: "Alessandro Luongo" <alessandro@luongo.pro>
发送时间: 2025-02-23 12:49:05 (星期日)
收件人: "Tiesunlong Shen" <tensorshen@mail.ynu.edu.cn>
主题: Re: Introduction from Prof. Erik Cambria - Tiesunlong Shen (Tensorlong)

Hello Tiesunlong
I hope this email find you well.

Can you remember me what is your plan in April? If I remember correctly you are a PhD student, finishing next year. 

So you can only do an internship of ~6 months, and you are not available for a research assistant position of 1 year, right?

Best,
Ale.






On Mon, 10 Feb 2025 at 09:29, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:
Hi Dr. Ale,


Looking forward to our meeting at 10:30 am. In case you haven't created a meeting link yet, here's my Zoom room:


https://us05web.zoom.us/j/81147383214?pwd=vU9RjyDyFJZdepopgTz1OXHXsRTcpf.1
Meeting ID: 811 4738 3214
Passcode: 9ctcny


Best regards,
Tiesunlong



-----原始邮件-----
发件人: "Alessandro Luongo" <alessandro@luongo.pro>
发送时间: 2025-02-07 22:11:30 (星期五)

收件人: "Tiesunlong Shen" <tensorshen@mail.ynu.edu.cn>
主题: Re: Introduction from Prof. Erik Cambria - Tiesunlong Shen (Tensorlong)

Tiesunlong,

Looking forward to it. 

Ale. 

On Fri, 7 Feb 2025 at 17:03, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:
Dear Dr. Ale,

I'm very glad you're feeling better! Monday 10:30 am works perfectly for me.

Regarding PyTorch - yes, I'm very familiar with it. In fact, it's my favorite deep learning framework, and I've used it extensively in all my research work. For graph-related libraries, I'm
well-versed in both PyG and DGL, which I've utilized in my graph-based projects. I'm also quite comfortable with the Transformers library from Hugging Face, which I've used extensively in my
LLM-based work.

I'm confident in my ability to handle most programming tasks, especially in this rapidly evolving era of LLMs. I'm particularly experienced with AI-assisted coding and always enthusiastic about
exploring new technologies, even without immediate benefits.

I should mention that I have a limitation - my spoken English and listening skills need improvement, as I've never lived outside China. While I'm comfortable with reading and writing in English,
my speaking and listening abilities are not as strong. However, I believe I could adapt relatively quickly after 1-2 months in an English-speaking environment.

I wanted to be upfront about both my strengths and limitations. In any case, I'm really glad you're feeling better now. Looking forward to our meeting on Monday!

Best regards,
Tiesunlong



-----原始邮件-----
发件人: "Alessandro Luongo" <alessandro@luongo.pro>
发送时间: 2025-02-07 11:15:00 (星期五)
收件人: "Tiesunlong Shen" <tensorshen@mail.ynu.edu.cn>
主题: Re: Introduction from Prof. Erik Cambria - Tiesunlong Shen (Tensorlong)

Dear Tiesunlong,
I am back alive. :) 
 
What about Monday morning at 10:30 ?
Do you know how to use pytorch? What's your favourite framework? 

Ale.





On Wed, 5 Feb 2025 at 09:50, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:
Dear Dr.Alessandro,

I'm sorry to hear you're not feeling well. I hope you'll feel better soon.

I'm flexible with the timing - please feel free to suggest any time that works best for you.

Take care,
Tiesunlong


-----原始邮件-----
发件人: "Alessandro Luongo" <alessandro@luongo.pro>
发送时间: 2025-02-05 09:01:00 (星期三)
收件人: "Tiesunlong Shen" <tensorshen@mail.ynu.edu.cn>
主题: Re: Introduction from Prof. Erik Cambria - Tiesunlong Shen (Tensorlong)

Dear Tiesunlong,

Unfortunately, I'm not feeling well and need to reschedule our call.

Could we find a time later today or tomorrow? I apologize for the inconvenience.

Best,
Alessandro

On Mon, 3 Feb 2025 at 17:45, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:
Dear Dr.Ale,

Perfect! I'll be ready for our Zoom call on Wednesday at 10:30 am (SG/CN Time). Looking forward to our conversation :)

Best regards,
Tiesunlong


-----原始邮件-----
发件人: "Alessandro Luongo" <alessandro@luongo.pro>
发送时间: 2025-02-03 08:00:00 (星期一)

收件人: "Tiesunlong Shen" <tensorshen@mail.ynu.edu.cn>
主题: Re: Introduction from Prof. Erik Cambria - Tiesunlong Shen (Tensorlong)

Tiesunlong,
Nice!
Let's talk Wednesday at 10:30 am on zoom! 

Best regards,
Ale. 

On Sun, 2 Feb 2025 at 16:45, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:
Dear Dr. Luongo,


Thank you for your kind reply. I am delighted to have the opportunity to speak with you.

Wednesday morning works perfectly for me. Please let me know the specific time and preferred platform (Zoom/Teams/etc.) that suits

you best - I'll make sure to be available.

Looking forward to our conversation.



Best regards,

Tiesunlong



-----原始邮件-----
发件人: "Alessandro Luongo" <alessandro@luongo.pro>
发送时间: 2025-02-02 14:42:31 (星期日)
收件人: "Tiesunlong Shen" <tensorshen@mail.ynu.edu.cn>
主题: Re: Introduction from Prof. Erik Cambria - Tiesunlong Shen (Tensorlong)

Hello Tiesunlong,

It's very nice to meet you. 
This position is really marginally related to QML. 
Are you available Tuesday or Wednesday morning for a call? 

Best,
Ale. 

On Sun, 2 Feb 2025 at 00:33, Tiesunlong Shen <tensorshen@mail.ynu.edu.cn> wrote:
Dear Dr. Luongo,

I hope this email finds you well. I am Tiesunlong Shen, a third-year PhD student from Yunnan University, China, writing to you upon Professor Erik Cambria's kind introduction.

My current research focuses on integrating graph structures into deep learning models, particularly in LLMs and diffusion models.
I have attached my CV & recent papers for your reference.

I recently read your excellent book "Quantum algorithms for data analysis" which sparked my interest in Quantum Machine Learning.
While the quantum computing aspects still feel quite foreign to my current expertise, I found particular resonance with the
classical machine learning and Monte Carlo components discussed. As you mentioned, one can build significantly upon the basic
axioms of quantum mechanics without diving too deep into the quantum mechanical details.

Although my background might not perfectly align with quantum computing, I am genuinely curious about potential intersections
between our research areas. I would greatly appreciate the opportunity to discuss possible collaboration or simply exchange ideas,
even if just to explore where our research interests might complement each other.

Thank you for your time and consideration.

Best regards,
Tiesunlong Shen