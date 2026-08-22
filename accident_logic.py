import math

vehicle_history = {}


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    intersection = width * height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


def update_vehicle(vehicle_id, center):

    if vehicle_id not in vehicle_history:

        vehicle_history[vehicle_id] = {
            "positions": [],
            "movements": [],
            "directions": []
        }

    data = vehicle_history[vehicle_id]

    movement = 0
    direction = (0, 0)

    if len(data["positions"]) > 0:

        previous = data["positions"][-1]

        dx = center[0] - previous[0]
        dy = center[1] - previous[1]

        movement = distance(previous, center)

        direction = (dx, dy)

    data["positions"].append(center)
    data["movements"].append(movement)
    data["directions"].append(direction)

    # Keep recent history
    data["positions"] = data["positions"][-20:]
    data["movements"] = data["movements"][-20:]
    data["directions"] = data["directions"][-20:]

    return movement


def sudden_stop(vehicle_id):

    if vehicle_id not in vehicle_history:
        return False

    movements = vehicle_history[vehicle_id]["movements"]

    if len(movements) < 8:
        return False

    previous = movements[-8:-3]
    recent = movements[-3:]

    previous_avg = sum(previous) / len(previous)
    recent_avg = sum(recent) / len(recent)

    return previous_avg > 5 and recent_avg < 2


def sudden_direction_change(vehicle_id):

    if vehicle_id not in vehicle_history:
        return False

    directions = vehicle_history[vehicle_id]["directions"]

    if len(directions) < 6:
        return False

    old_dx, old_dy = directions[-5]
    new_dx, new_dy = directions[-1]

    old_magnitude = math.sqrt(old_dx ** 2 + old_dy ** 2)
    new_magnitude = math.sqrt(new_dx ** 2 + new_dy ** 2)

    if old_magnitude < 3 or new_magnitude < 3:
        return False

    dot = old_dx * new_dx + old_dy * new_dy

    cosine = dot / (
        old_magnitude * new_magnitude
    )

    # Large direction change
    return cosine < 0.5


def collision_score(vehicles):

    score = 0
    collision_pairs = []

    ids = list(vehicles.keys())

    for i in range(len(ids)):

        for j in range(i + 1, len(ids)):

            id1 = ids[i]
            id2 = ids[j]

            v1 = vehicles[id1]
            v2 = vehicles[id2]

            center1 = v1["center"]
            center2 = v2["center"]

            box1 = v1["box"]
            box2 = v2["box"]

            d = distance(center1, center2)

            iou = calculate_iou(box1, box2)

            pair_score = 0

            # -------------------------
            # Collision proximity
            # -------------------------

            if iou > 0.10:
                pair_score += 30

            elif d < 60:
                pair_score += 15

            # -------------------------
            # Sudden stop
            # -------------------------

            if sudden_stop(id1):
                pair_score += 20

            if sudden_stop(id2):
                pair_score += 20

            # -------------------------
            # Sudden direction change
            # -------------------------

            if sudden_direction_change(id1):
                pair_score += 15

            if sudden_direction_change(id2):
                pair_score += 15

            # -------------------------
            # Register suspicious pair
            # -------------------------

            if pair_score >= 30:

                collision_pairs.append(
                    (id1, id2)
                )

                score += pair_score

    return min(score, 100), collision_pairs