# Data Management

The data management module is responsible for handling all data-related tasks within the project, including retrieval, processing, and storage. The primary goal is to automate the processes of data acquisition, cleaning, and processing as much as possible.

To achieve this automation, **Apache Airflow** is utilized. You can find the implementation of the data management pipeline in the following repository:


[HUBer: Data Management Pipeline](https://github.com/sonyaarom/huber_airflow_dm)

`main.py` of this folder is created solely for the demonstration purposes. The `main.py` showcases how the data is being extracted from the sitemap of the website and how the HTML content is being retrieved from the website. 

The next step in the workflow involves applying processing steps such as chunking, embedding, and storing the data in a vector database. Detailed explanations of the different chunking methods, their implementation, and evaluations can be found in the following folder:


[Chunking methods](https://github.com/sonyaarom/huber_bot/tree/main/src/data_chunker)


