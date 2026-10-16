# User proposal

Original proposal (verbatim):

> btw somethign that came up is maybe combining the new sleep stuff with someone else, maybe instead of a single window, we can combine "Active" regions by usign them as windows, until they merge, so like we start with a single window of active (not sleeping) then if we commit somewhere else then we add that as well as an active regon, so like computation starts "effecient" and then expands, but i wonder if stuff like that alr exist

Clarification (verbatim):

> well nothing you mentioned really touches my idea, also window diffusion might eb the closest actually btu its again just 1 window, and yeah we need periodic refreshers to allow more regions to open

Implementation request (verbatim):

> do you wanna try and implement this? take inspiartion or START from window fiffusion work

Working hypothesis: several active computation windows may follow distant
decoding progress without recomputing the entire interval between them. Refreshes
provide global discovery. The first implementation tests this mechanism; it does
not establish that commitment proximity predicts context safety, novelty, or an
advantage over Window-Diffusion. Engineering choices are listed in README.md.
