# Feedback

## Things I found confusing

**Data structure:** It took me a while to figure out that the 4 channels are stored sequentially in the flat vector (all R values first, then all G, etc.) rather than interleaved. Once I understood that, tasks 2.3 and 2.4 were straightforward, but I think a short comment in the notebook would save some time.

**Train/test overlap:** The RF notebook samples train and test indices independently per class, which means the same sample can end up in both. I noticed this when reading the code carefully (task 2.1 question). The CNN notebook avoids this by permuting first and then slicing — I think that approach is cleaner and it would be worth updating the RF notebook to match.

**One-hot vs integer labels:** The RF's `fit()` accepts one-hot labels directly without complaint. The CNN's CrossEntropyLoss needs integer class indices. I had to parse them via `np.where(label==1)[0][0]` in `__getitem__`. Took me a moment to figure out why the RF worked fine but the CNN gave a shape error.

**Training time:** Running the CNN on CPU is slow. 10 epochs took around 10–15 minutes on my machine. I'd recommend noting in the notebook that Colab with GPU runtime is much faster for this.

## Suggestions

- A `requirements.txt` would be helpful to include in the base repo (I added one in my fork).
- A one-line comment explaining the channel ordering (R: 0–783, G: 784–1567, ...) somewhere near the data loading cell would save time for people doing tasks 2.3/2.4.
