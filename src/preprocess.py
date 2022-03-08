import cv2
import numpy as np


def reverse_vector(vector):
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    RCLAV = 23
    # LCLAV = 23

    p1 = RLUNG*2
    p2 = p1 + LLUNG*2
    p3 = p2 + HEART*2
    p4 = p3 + RCLAV*2

    rl = vector[:p1].reshape(-1, 2)
    ll = vector[p1:p2].reshape(-1, 2)
    h = vector[p2:p3].reshape(-1, 2)
    rc = vector[p3:p4].reshape(-1, 2)
    lc = vector[p4:].reshape(-1, 2)

    return rl, ll, h, rc, lc


def draw_binary(img, organ, color):
    contorno = organ.reshape(-1, 1, 2)

    contorno = contorno.astype('int')

    img = cv2.drawContours(img, [contorno], -1, color, -1)

    return img


def get_seg(landmarks):
    leftlung, rightlung, heart, rc, lc = reverse_vector(landmarks.reshape(-1))

    raw = np.zeros([1024, 1024])

    raw = draw_binary(raw, leftlung, 1)
    raw = draw_binary(raw, rightlung, 1)

    raw = draw_binary(raw, heart, 2)

    raw = draw_binary(raw, rc, 3)
    raw = draw_binary(raw, lc, 3)

    return raw
