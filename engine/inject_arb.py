import json
import time
from confluent_kafka import Producer

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def main():
    producer = Producer({'bootstrap.servers': 'localhost:19092'})
    topic = 'ticks'

    # Cycle: USD -> EUR -> JPY -> USD
    # USD -> EUR: 0.9 (EUR_USD ask = 1.111111)
    # EUR -> JPY: 160.0 (EUR_JPY bid = 160.0)
    # JPY -> USD: 0.0075 (USD_JPY ask = 133.333333)

    ticks = [
        {
            "type": "PRICE",
            "time": "2023-10-27T10:00:00Z",
            "instrument": "EUR_USD",
            "closeoutBid": "1.0000", # Irrelevant
            "closeoutAsk": "1.111111", # USD -> EUR rate = 0.9
            "status": "tradeable"
        },
        {
            "type": "PRICE",
            "time": "2023-10-27T10:00:01Z",
            "instrument": "EUR_JPY",
            "closeoutBid": "160.0", # EUR -> JPY rate = 160.0
            "closeoutAsk": "161.0", # Irrelevant
            "status": "tradeable"
        },
        {
            "type": "PRICE",
            "time": "2023-10-27T10:00:02Z",
            "instrument": "USD_JPY",
            "closeoutBid": "130.0", # Irrelevant
            "closeoutAsk": "133.333333", # JPY -> USD rate = 0.0075
            "status": "tradeable"
        }
    ]

    print("Injecting arbitrage ticks...")
    for tick in ticks:
        producer.produce(
            topic,
            key=tick['instrument'],
            value=json.dumps(tick),
            callback=delivery_report
        )
        producer.poll(0)
        time.sleep(0.5) # Small delay to ensure processing order

    producer.flush()
    print("Injection complete.")

if __name__ == "__main__":
    main()
