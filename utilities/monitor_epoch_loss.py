#!/usr/bin/env python3
import time
import csv
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import matplotlib.dates as mdates
import argparse

def main(
        epochLossFilepath,
        maximumNumberOfPoints,
        updateInterval
):

    """def parse_time(s):
        s = s.strip()
        if TIME_FORMAT == "float":
            return float(s)
        # ISO ou autre format lisible par datetime.fromisoformat
        try:
            return datetime.fromisoformat(s)
        except Exception:
            # fallback : timestamp (float seconds)
            return datetime.fromtimestamp(float(s))
    """

    # Stockage des données
    xs = deque(maxlen=maximumNumberOfPoints)
    ys = deque(maxlen=maximumNumberOfPoints)

    fig, ax = plt.subplots()
    (line_plot,) = ax.plot([], [], "-o", markersize=4, label="value")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.legend()

    # Si on utilise des datetimes pour l'axe x, configurer le formateur
    """use_datetime_x = (TIME_FORMAT != "float")
    if use_datetime_x:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()
    """

    # Ouvrir le fichier et lire l'historique
    f = open(epochLossFilepath, "r", newline="")
    # lire l'en-tête (optionnel)
    header = f.readline()
    #print(f"header = {header}")
    # lire les lignes existantes
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            t = int(parts[0])
            y = float(parts[1])
        except Exception:
            continue
        xs.append(t)
        ys.append(y)
    #print(f"ys = {ys}")
    # position du curseur est à la fin du fichier après la lecture
    # fonction d'update appelée par FuncAnimation
    def update(frame):
        # lire toutes les nouvelles lignes (non-bloquant)
        new_lines = f.readlines()
        if not new_lines:
            # rien de nouveau
            return (line_plot,)

        changed = False
        for raw in new_lines:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                t = int(parts[0])
                y = float(parts[1])
            except Exception:
                continue
            xs.append(t)
            ys.append(y)
            changed = True

        if not changed:
            return (line_plot,)

        # Mettre à jour la courbe
        """if use_datetime_x:
            line_plot.set_data(mdates.date2num(list(xs)), list(ys))
            ax.relim()
            ax.autoscale_view()
            # garder formatteur
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        else:
        """
        line_plot.set_data(list(xs), list(ys))
        ax.relim()
        ax.autoscale_view()

        return (line_plot,)

    ani = animation.FuncAnimation(fig, update, interval=updateInterval, blit=False)

    try:
        plt.show()
    finally:
        f.close()

if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('epochLossFilepath', help="The epoch loss filepath")
        parser.add_argument('--maximumNumberOfPoints', help="The maximum number of points to display. Default=1000", type=int, default=1000)
        parser.add_argument('--updateInterval', help="The update interval in ms. Default: 1000", type=int, default=1000)
        args = parser.parse_args()
    main(
        args.epochLossFilepath,
        args.maximumNumberOfPoints,
        args.updateInterval
    )