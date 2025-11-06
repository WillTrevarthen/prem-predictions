from operator import index
import pandas as pd
import pickle


def main():
    df = pd.read_csv('epl-2025-GMTStandardTime.csv')
    # Split into new columns
    df[['Date', 'Time']] = df['Date'].str.split(' ', expand=True)
    df = df[['Date', 'Time', 'Home Team', 'Away Team']]
    df.to_csv('inference.csv', index=False)


if __name__ == '__main__':
    main()
