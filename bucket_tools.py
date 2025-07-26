bucket_options = {
     320: [
        (160, 288),
        (176, 304),
        (192, 320),
        (208, 320),
        (224, 320),
        (240, 304),
        (256, 288),
        (272, 272),
        (288, 256),
        (304, 240),
        (320, 224),
    ],
    480: [
        (320, 448),
        (336, 432),
        (352, 416),
        (368, 400),
        (384, 384),
        (400, 368),
        (416, 352),
        (432, 336),
        (448, 320),
    ],
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ],
}


def find_nearest_bucket(h, w, resolution=640):
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket

