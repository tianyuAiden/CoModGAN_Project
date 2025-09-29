"""
mask generator

"""
import numpy as np
import cv2
import random
import torch


def generate_brush_strokes(h, w):
    mask = np.zeros((h, w), np.uint8)
    num_brush = random.randint(0, 20)

    for _ in range(num_brush):
        num_vertices = random.randint(4, 18)  # 顶点数
        angle = random.uniform(0, 2 * np.pi)
        brush_w = random.randint(12, 48)

        start_x = random.randint(0, w)
        start_y = random.randint(0, h)
        vertices = []

        for i in range(num_vertices):
            dx = int(np.cos(angle) * random.randint(20, 100))
            dy = int(np.sin(angle) * random.randint(20, 100))
            end_x = np.clip(start_x + dx, 0, w - 1)
            end_y = np.clip(start_y + dy, 0, h - 1)
            vertices.append((end_x, end_y))
            start_x, start_y = end_x, end_y
            angle += random.uniform(-np.pi / 4, np.pi / 4)

        for i in range(len(vertices) - 1):
            cv2.line(mask, vertices[i], vertices[i + 1], 1, brush_w)

    return mask


def generator_rectangles(h, w):
    """ generate a random rectangle """
    mask = np.zeros((h, w), np.uint8)

    for _ in range(random.randint(0, 5)):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        x2 = random.randint(x1, w)
        y2 = random.randint(y1, h)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

    for _ in range(random.randint(0, 10)):
        rw, rh = random.randint(10, w // 2), random.randint(10, h // 2)
        x1 = random.randint(0, w - rw)
        y1 = random.randint(0, h - rh)
        cv2.rectangle(mask, (x1, y1), (x1 + rw, y1 + rh), 1, -1)

    return mask


def generator_freeform_mask(h=512, w=512):
    mask_brush = generate_brush_strokes(h, w)
    mask_rect = generator_rectangles(h, w)
    mask = np.clip(mask_brush + mask_rect, 0, 1)

    #  1×H×W
    return torch.from_numpy(mask).unsqueeze(0).float()
