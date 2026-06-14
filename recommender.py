from pprint import pprint

from cineiq import train_and_save


if __name__ == "__main__":
    artifacts = train_and_save()
    print("Hybrid engine trained successfully.")
    print(
        f"Movies: {len(artifacts['movies'])}, Users: {len(artifacts['user_ids'])}, "
        f"Vector Features: {artifacts['vector_features']}"
    )
    pprint(
        {
            "datasets": artifacts["dataset_summary"],
            "sentiment_source": artifacts["sentiment_source"],
            "tech_stack": artifacts["tech_stack"],
            "default_weights": artifacts["default_weights"],
        }
    )
