from matplotlib import pyplot as plt
import json
import os.path as osp

def visualize(json_path, image_folder):
    with open(json_path, 'r') as f:
        results = json.load(f)

    for idx, query_path in enumerate(results.keys()):
        query_image = plt.imread(osp.join(image_folder, 'query_a', query_path))

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(query_image)
        ax.set_title('query')

        #print('Top 10 images are as follow:')

        for i in range(10):
            gallery_path = results[query_path][i]
            img_path = osp.join(image_folder, 'gallery_a', gallery_path)
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig("show_%d.jpg" % idx)

        if idx >= 10:
            break
        #print('result saved to show.')

if __name__ == '__main__':
    json_path = 'submit/reid_mgn_resnet101_ibn_20191225_084306_flip_rerank_cross.json'
    image_folder = '/Volumes/Data/比赛/行人重识别2019/data/复赛/测试集A'
    visualize(json_path, image_folder)


