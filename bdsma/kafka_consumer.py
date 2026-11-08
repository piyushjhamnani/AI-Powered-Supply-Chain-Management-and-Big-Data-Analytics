"""Optional Kafka consumer that stores received events as JSON Lines.

Run:
    python bdsma/kafka_consumer.py --bootstrap localhost:9092 --output data/kafka_captured_events.jsonl
"""
import argparse
import json
from kafka import KafkaConsumer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="supply-chain-events")
    parser.add_argument("--output", default="data/kafka_captured_events.jsonl")
    parser.add_argument("--max-events", type=int, default=0, help="0 means keep listening")
    args = parser.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        consumer_timeout_ms=1000 if args.max_events else None,
    )
    count = 0
    try:
        with open(args.output, "a", encoding="utf-8") as output:
            for message in consumer:
                output.write(json.dumps(message.value, ensure_ascii=False) + "\n")
                output.flush()
                count += 1
                if args.max_events and count >= args.max_events:
                    break
    except KeyboardInterrupt:
        print("Consumer stopped by user.")
    finally:
        consumer.close()
    print(f"Captured {count} events in {args.output}.")


if __name__ == "__main__":
    main()
