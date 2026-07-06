# Training data
xs = [10, 20, 30, 40]
ys = [20, 40, 60, 80]

# Initial weights
w = 0.0
b = 0.0

lr = 0.001
epoch = 0

while True:
    command = input("\nPress Enter to run 1 epoch | q = quit: ")

    if command.lower() == "q":
        break

    print("\n" + "=" * 70)
    print("EPOCH:", epoch)
    print("=" * 70)

    total_loss = 0

    print("Model at the beginning of the epoch:")
    print(f"y_pred = {w:.6f} * x + {b:.6f}")
    print("-" * 70)

    for i, (x, y) in enumerate(zip(xs, ys)):
        print(f"\nData sample {i + 1}")
        print("-" * 30)

        # Store old w and b so we can see before and after the update
        old_w = w
        old_b = b

        # 1) Prediction
        y_pred = w * x + b

        print("1) Prediction calculation:")
        print("y_pred = w * x + b")
        print(f"y_pred = {w:.6f} * {x} + {b:.6f}")
        print(f"y_pred = {y_pred:.6f}")

        # 2) Error
        error = y - y_pred

        print("\n2) Error calculation:")
        print("error = y - y_pred")
        print(f"error = {y} - {y_pred:.6f}")
        print(f"error = {error:.6f}")

        # 3) Loss
        loss = error ** 2
        total_loss += loss

        print("\n3) Loss calculation:")
        print("loss = error ** 2")
        print(f"loss = ({error:.6f}) ** 2")
        print(f"loss = {loss:.6f}")

        # 4) Gradient
        dL_dw = -2 * x * error
        dL_db = -2 * error

        print("\n4) Gradient calculation:")
        print("dL/dw = -2 * x * error")
        print(f"dL/dw = -2 * {x} * {error:.6f}")
        print(f"dL/dw = {dL_dw:.6f}")

        print("\ndL/db = -2 * error")
        print(f"dL/db = -2 * {error:.6f}")
        print(f"dL/db = {dL_db:.6f}")

        # 5) Update
        w = w - lr * dL_dw
        b = b - lr * dL_db

        print("\n5) Parameter update:")
        print("w_new = w_old - lr * dL/dw")
        print(f"w_new = {old_w:.6f} - {lr} * ({dL_dw:.6f})")
        print(f"w_new = {w:.6f}")

        print("\nb_new = b_old - lr * dL/db")
        print(f"b_new = {old_b:.6f} - {lr} * ({dL_db:.6f})")
        print(f"b_new = {b:.6f}")

        print("\nModel after this data sample:")
        print(f"y_pred = {w:.6f} * x + {b:.6f}")

    print("\n" + "-" * 70)
    print("Epoch finished.")
    print(f"Total loss = {total_loss:.6f}")
    print(f"w at the end of the epoch = {w:.6f}")
    print(f"b at the end of the epoch = {b:.6f}")
    print(f"Model at the end of the epoch: y_pred = {w:.6f} * x + {b:.6f}")

    x_test = 50
    test_pred = w * x_test + b

    print("\nTest:")
    print(f"x_test = {x_test}")
    print(f"prediction = {w:.6f} * {x_test} + {b:.6f}")
    print(f"prediction = {test_pred:.6f}")

    epoch += 1

print("\nFinal model:")
print(f"y = {w:.6f} * x + {b:.6f}")

