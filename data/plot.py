import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MONITOR_PATH = "monitor.csv"
OUT_PLOT = "learning_curve.png"
WINDOW_SIZE = 100 # Number of episodes to average over

def main():
    df = pd.read_csv(MONITOR_PATH, skiprows=1)
    df["episode"] = range(1, len(df) + 1)

    df["rolling_mean"] = df["r"].rolling(
        window=WINDOW_SIZE, 
        min_periods=1, 
        center=True
    ).mean()

 
    x = df["episode"].values
    y = df["rolling_mean"].values
    

    valid_indices = ~np.isnan(y)
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
   
    p = np.polyfit(x_valid, y_valid, 1)
    
    poly_func = np.poly1d(p)
    
    df["linear_trend"] = poly_func(df["episode"])


    plt.figure(figsize=(10, 5))
    
    plt.plot(
        df["episode"], 
        df["r"], 
        label="Raw Episode Return (r)", 
        alpha=0.3, # Made more transparent to focus on the trend
        color='blue'
    )
    
    plt.plot(
        df["episode"], 
        df["rolling_mean"], 
        label=f"{WINDOW_SIZE}-Episode Moving Average", 
        color='red', 
        linewidth=2
    )

    plt.plot(
        df["episode"], 
        df["linear_trend"], 
        label="Overall Linear Trend", 
        color='green', 
        linestyle='--', # Dotted line
        linewidth=2
    )
    

    plt.ylim(bottom=-200) 
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curve – Trend Analysis (Raw, Rolling Mean, and Linear Fit)")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_PLOT)
    plt.close()

    print("Learning curve saved to:", OUT_PLOT)

if __name__ == "__main__":
    main()