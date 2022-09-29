# Dalle-Mini-RL
Fine-tuning Dalle-Mini with RL to not produce NSFW images

## High-level Overview

My final project was to train [DALL⋅E-Mini](https://github.com/borisdayma/dalle-mini) to not output NSFW images using RL. The reward signal was primarily provided from a [NSFW image classifier](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) that was trained to read CLIP embeddings of images and output a probability of them being NSFW. The motivation for this project was to see how transferable RLHF was to text-to-image models, since its effectiveness for fine-tuning language models has already been amply demonstrated. There was no human feedback component to the training of this model, so this is more of a test-bed of how well Dalle-Mini can be fine-tuned with RL against a classifier more than how well it can be trained to match human preferences for/against NSFW images. 

## Approach

First, the DALL⋅E-Mini architecture had to be modified so that it could be fine-tuned with PPO. This basically looked like modifying the code from [minDALL⋅E](https://github.com/kuprel/min-dalle) so that the architecture could utilize a value head. This would be used by the critic component of the PPO actor-critic agent to output predictions for the value of the state the model was in. 

Next, the NSFW classifier had to be converted from Keras to PyTorch. I also had to develop a reward signal based on the classifier’s outputs. I decided to use some human judgment and err on the side of caution - I’d strongly penalize any image with a probability greater than 0.4 of being NSFW and weakly penalize anything between 0.1 and 0.4 probability of being NSFW. Thinking of this as a high-stakes reliability problem, I considered it to be much worse to output a NSFW image than to output a bad SFW image, so it made sense to penalize even for borderline cases. However, to not completely obliterate the model’s performance competitiveness, I included a component of the reward based on the similarity between the CLIP text embeddings of the caption and image embeddings of the output image. This component only kicked in for (presumed) SFW images, with the idea being that it was fine to “lose track of reality” for NSFW images (i.e. not check how well the model was generating an image relevant to the caption), but obviously not fine to do so for SFW images. On my first training run, I didn’t include this component, leading to fairly nonsensical results - the model definitely didn’t output NSFW images, but it basically just outputted noise for any input caption, which was rewarded highly by the classifier.

For the PPO setup, I utilized some old code from [a previous RL(HF?) project](https://github.com/anshradh/trl_custom) and from the [trl repo](https://github.com/lvwerra/trl) but made some modifications so that the PPO would use distinct actor and critic networks (so that updates to one would not interfere with the other). I also made various other changes/optimizations to deal with the different nature of the task (text-to-image instead of text-to-text) and incorporated some other specific PPO implementation details that those previous projects didn’t have.

The dataset I used was [laion400m](https://huggingface.co/datasets/laion/laion400m), which conveniently contained NSFW labels for 400 million images as well as the captions for the images. I didn’t actually use the reference images themselves, just their captions to prompt the policy model and the reference original model. I probably should have first fine-tuned the model on the SFW images from the dataset first, but I skipped this step since the model already seemed to be noticeably better at generating SFW images than NSFW ones, probably just as a function of its original training dataset.
Training was very slow - this was likely a big bottleneck to achieving better results and I probably should have trained with at least data parallelism.

## Results

The results I got were not extremely promising, but I think there’s still evidence that this kind of fine-tuning can work on text-to-image models. On a mixed sample of 50 images from the dataset, the model achieved a mean reward of 0.43 compared to the reference model’s reward of 0.29, though the median rewards were less promising (3.59 for the reference, 3.13 for the trained policy). From analyzing more facets of both distributions, it seems like the model sacrificed some performance competitiveness on SFW images but did slightly better in terms of not outputting NSFW images (or not as NSFW ones as the reference did) - conditioning on negative reward, the policy model had higher mean (-18.57 reward instead of -21.43 reward).

Here are the summary statistics on the entire 50-sample batch, where before refers to the reference model and after refers to the trained policy model.

![stats](eval_stats.jpeg)

Subjectively, the model does seem to output slightly less NSFW images in comparison to the reference model (on captions that lead to NSFW-ish images):

![nsfw_comps](NSFWImageComps.jpeg)

And the model isn’t obviously worse at some more normal captions:

![good_sfw_comps](SFWImageComps.jpeg)

Though it is pretty obviously worse on some others…

![bad_sfw_comps](MoreSFWImageComps.jpeg)

## Retrospective and Future Directions

Mostly, these results demonstrated to me the importance of using a robust and on-policy reward model for the reward signal when performing RLHF. It seems likely to me that the reward model was just poorly calibrated to the skill level of the policy model and thus struggled to provide meaningful guidance for optimization of the policy. In future projects where I attempt to fine-tune models with RL, I will definitely spend more time making sure the reward model is appropriately trained (and likely will train my own, ideally on human feedback on the policy model’s outputs, like proper RLHF).

Throughout my several training runs, PPO did generally fine at optimizing reward, but this didn’t always line up with the desired outcome. Like I said above, I think there’s definitely something to the idea of fine-tuning text-to-image models with RL, but I think to really test this I’d need to develop a faster training procedure, train my own reward model from scratch on human feedback data, and likely just get generally better at RL :P.


