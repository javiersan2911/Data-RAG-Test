import os

import dotenv
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

FAQ_CSV_PATH = "split_files/train_counsel_chat.csv"
FAQ_CHROMA_PATH = "vector"

dotenv.load_dotenv()


def load_vectordb_from_csv() -> None:
    """Loads a vector database from a CSV file.

    Returns:
        None

    """
    loader = CSVLoader(file_path=FAQ_CSV_PATH, source_column="answerText", encoding="utf8")
    answers = loader.load()

    Chroma.from_documents(
        answers, OpenAIEmbeddings(), persist_directory=FAQ_CHROMA_PATH
    )


def split_csv_data() -> None:
    """
    Split the Counsel Chat CSV file into separate files based on the 'split' field.

    Args:
        None

    Returns:
        None

    """
    # Read the CSV file
    df = pd.read_csv('counsel_chat.csv')

    # Get the unique values of the 'split' field
    split_values = df['split'].unique()

    # Create the directory to store the split files
    if not os.path.exists('split_files'):
        os.mkdir('split_files')

    # Split the CSV file and save it to separate files
    for split_value in split_values:
        # Filter the data based on the split value
        split_df = df[df['split'] == split_value]

        # Save the filtered data to a separate file
        split_df.to_csv(f'split_files/{split_value}_counsel_chat.csv', index=False)


if __name__ == "__main__":
    split_csv_data()
    load_vectordb_from_csv()
