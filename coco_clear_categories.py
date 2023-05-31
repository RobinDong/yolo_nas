import os

parent_dir = "/media/data2/sanbai/datasets/coco/"


def extract_categories(label_file: str, img_file: str):
    lines = []
    filename = parent_dir + label_file[2:]
    if not os.path.exists(filename):
        return lines
    file_lines = []
    with open(filename, "r") as fp:
        for line in fp:
            cat = int(line.split()[0])
            if cat in {14, 15, 16, 80}:
                file_lines.append(line)
    if len(file_lines) <= 0:
        print("remove:", label_file[2:])
        try:
            os.remove(parent_dir + label_file[2:])
            os.remove(parent_dir + img_file[2:])
        except OSError:
            pass
    lines += file_lines

    try:
        os.remove(filename)
    except OSError:
        pass
    if len(lines) > 0:
        with open(filename, "w") as fp:
            for line in lines:
                fp.write(line.strip() + "\n")
    return lines


for index_file in ["train2017.txt", "val2017.txt"]:
    lines = []
    with open(parent_dir + index_file, "r") as fp:
        for line in fp:
            line = line.strip()
            label_file = line.replace(
                "images", "labels"
            ).replace(
                ".jpg", ".txt"
            )
            array = extract_categories(label_file, line)
            if array:
                lines.append(line)

    with open(parent_dir + index_file + ".new", "w") as fp:
        for line in lines:
            fp.write(line + "\n")
