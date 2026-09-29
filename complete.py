import math


# -------------------------
# DATA
# -------------------------

x = 1.0
target = 1.0


# -------------------------
# INITIAL PARAMETERS
# -------------------------

w1 = 0.5
b1 = 0.0

w2 = 0.5
b2 = 0.0

learning_rate = 0.1


# -------------------------
# SIGMOID
# -------------------------

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# -------------------------
# TRAINING
# -------------------------

for step in range(100):

    # =========================
    # FORWARD PASS
    # =========================

    z_hidden = x * w1 + b1

    h = sigmoid(z_hidden)

    z_output = h * w2 + b2

    y = sigmoid(z_output)


    # =========================
    # LOSS
    # =========================

    loss = 0.5 * (y - target) ** 2


    # =========================
    # BACKWARD PASS
    # =========================

    # dL/dy
    dL_dy = y - target

    # dy/dz_output
    dy_dz_output = y * (1 - y)

    # dz_output/dw2
    dz_output_dw2 = h

    # dz_output/db2
    dz_output_db2 = 1

    # dL/dw2
    dL_dw2 = (
        dL_dy
        * dy_dz_output
        * dz_output_dw2
    )

    # dL/db2
    dL_db2 = (
        dL_dy
        * dy_dz_output
        * dz_output_db2
    )


    # -------------------------
    # HIDDEN LAYER
    # -------------------------

    # dz_output/dh
    dz_output_dh = w2

    # dh/dz_hidden
    dh_dz_hidden = h * (1 - h)

    # dz_hidden/dw1
    dz_hidden_dw1 = x

    # dz_hidden/db1
    dz_hidden_db1 = 1

    # dL/dw1
    dL_dw1 = (
        dL_dy
        * dy_dz_output
        * dz_output_dh
        * dh_dz_hidden
        * dz_hidden_dw1
    )

    # dL/db1
    dL_db1 = (
        dL_dy
        * dy_dz_output
        * dz_output_dh
        * dh_dz_hidden
        * dz_hidden_db1
    )


    # =========================
    # UPDATE PARAMETERS
    # =========================

    w1 = w1 - learning_rate * dL_dw1
    b1 = b1 - learning_rate * dL_db1

    w2 = w2 - learning_rate * dL_dw2
    b2 = b2 - learning_rate * dL_db2


    # =========================
    # PRINT
    # =========================

    if step % 10 == 0:
        print(
            f"step={step:3d} "
            f"loss={loss:.6f} "
            f"y={y:.6f}"
        )