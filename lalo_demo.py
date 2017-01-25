__author__ = 'cipriancorneanu'

from csdm.load_300w import load_300w
import cPickle
import matplotlib.pyplot as plt
import numpy as np

def augmenter(images, inits, cascade, n_augs):
    inits = np.tile(cascade._decode_parameters(inits), (n_augs, 1, 1))
    angles = np.random.uniform(low=-np.pi/4.0, high=np.pi/4.0, size=len(inits))
    disps = np.random.uniform(low=0.95, high=1.05, size=(len(inits), 2))
    scales = np.random.uniform(low=0.9, high=1.1, size=len(inits))
    mapping = np.tile(np.array(range(len(images)), dtype=np.int32), (n_augs,))
    for i in range(len(inits)):
        an, sc, dx, dy = angles[i], scales[i], disps[i][0], disps[i][1]
        mn = np.mean(inits[i, ...], axis=0)[None, :]
        inits[i, ...] = np.dot(
            inits[i, ...] - mn,
            sc * np.array([[np.cos(an), -np.sin(an)], [np.sin(an), np.cos(an)]], dtype=np.float32)
        ) + mn * [dx, dy]

    return cascade._encode_parameters(inits), mapping

def apply(mpath, images, steps=None):
    model = cPickle.load(open(mpath, 'rb'))
    # predictions = model.align(data['images'], num_steps=steps, save_all=True)
    predictions = model.align(images, num_steps=steps, save_all=True, augmenter=augmenter, n_augs=25)
    #cPickle.dump(predictions, open(rpath, 'wb'), cPickle.HIGHEST_PROTOCOL)
    return predictions

if __name__ == '__main__':
    # Load results and show
    model_file, results_file = ('continuous_300w.pkl', 'continuous_300w_results.pkl')
    path = '/Users/cipriancorneanu/Research/data/300w/'

    # Load image
    image = load_300w(path)['test']['images'][23:24]

    # Apply model
    prediction = apply(path+model_file, image, steps=3)
    step_1 = np.mean(prediction[0][0][0].reshape((25, 68, -1)), axis=0)
    step_2 = np.mean(prediction[0][1][0].reshape((25, 68, -1)), axis=0)
    step_3 = np.mean(prediction[0][2][0].reshape((25, 68, -1)), axis=0)

    # Show
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(np.squeeze(image))
    ax1.scatter(step_1[:,1], step_1[:,0])
    ax2.imshow(np.squeeze(image))
    ax2.scatter(step_2[:,1], step_2[:,0])
    ax3.imshow(np.squeeze(image))
    ax3.scatter(step_3[:,1], step_3[:,0])

    plt.savefig('./prediction.png')