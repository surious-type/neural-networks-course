from matplotlib import pyplot as plt


def plot_MAPE_optimizer(df):
    names = sorted(df["optimizer_class"].unique())

    values = [
        df.loc[df["optimizer_class"] == name, "mape"].dropna().values for name in names
    ]

    plt.figure(figsize=(12, 6))

    plt.boxplot(values, tick_labels=names, showfliers=False)

    plt.yscale("log")
    plt.xlabel("Оптимизатор")
    plt.ylabel("MAPE")
    plt.title("Сравнение качества оптимизаторов")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def plot_MAPE_batch(df, opimizer, loss):
    data = df[(df["optimizer_class"] == opimizer) & (df["model_loss"] == loss)]

    batch_mape = data.groupby("batch_size")["mape"].median()

    plt.figure(figsize=(9, 5))

    plt.plot(
        batch_mape.index,
        batch_mape.values,
        marker="o",
    )

    plt.yscale("log")
    plt.xticks(batch_mape.index)
    plt.xlabel("Размер батча")
    plt.ylabel("Медианная MAPE")
    plt.title("Влияние размера батча на качество обучения")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_MAPE_neurons(df, optimizer_class, model_loss, batch_size):
    data = df[
        (df["optimizer_class"] == optimizer_class)
        & (df["model_loss"] == model_loss)
        & (df["batch_size"] == batch_size)
    ]

    for func, group in data.groupby("func"):
        group = group.sort_values("neurons")
        plt.plot(group["neurons"], group["mape"], marker="o", label=func)

    plt.yscale("log")
    plt.xlabel("Количество нейронов")
    plt.ylabel("MAPE")
    plt.title(f"{optimizer_class} x {model_loss} x {batch_size}")
    plt.legend()
    plt.grid()
    plt.show()
