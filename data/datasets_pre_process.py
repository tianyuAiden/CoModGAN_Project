from torchvision import transforms
import torchvision.datasets as datasets
from mask_generator import generator_freeform_mask
import matplotlib.pyplot as plt

train_url = '../datasets/ffhq/train'
test_url = '../datasets/ffhq/test'

# dataset pre-process
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# masks generator

if __name__ == '__main__':
    for i in range(3):
        mask = generator_freeform_mask(512, 512)
        mask_np = mask.squeeze().numpy()

        plt.imshow(mask_np, cmap='gray')
        plt.title(f'test mask{i}')
        plt.axis('off')
        plt.show()
