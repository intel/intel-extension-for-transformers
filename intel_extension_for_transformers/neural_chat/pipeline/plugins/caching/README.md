# 🚀 What is caching plugin?

When LLM service encounters higher traffic levels, the expenses related to LLM API calls can become substantial. Additionally, LLM services might exhibit slow response times. Hence, we leverage GPTCache to build a semantic caching plugin for storing LLM responses. This README.md file provides an overview of the functionality of caching plugin, how to use it, and some example code snippets.

# 😎 What can this help with?

Caching plugin offers the following primary benefits:

- **Decreased expenses**: Caching plugin effectively minimizes expenses by caching query results, which in turn reduces the number of requests and tokens sent to the LLM service.
- **Enhanced performance**: Caching plugin can also provide superior query throughput compared to standard LLM services.
- **Improved scalability and availability**: Caching plugin can easily scale to accommodate an increasing volume of of queries, ensuring consistent performance as your application's user base expands.

# 🤔 How does it work?

Online services often exhibit data locality, with users frequently accessing popular or trending content. Cache systems take advantage of this behavior by storing commonly accessed data, which in turn reduces data retrieval time, improves response times, and eases the burden on backend servers. Traditional cache systems typically utilize an exact match between a new query and a cached query to determine if the requested content is available in the cache before fetching the data.

However, using an exact match approach for LLM caches is less effective due to the complexity and variability of LLM queries, resulting in a low cache hit rate. To address this issue, GPTCache adopt alternative strategies like semantic caching. Semantic caching identifies and stores similar or related queries, thereby increasing cache hit probability and enhancing overall caching efficiency. GPTCache employs embedding algorithms to convert queries into embeddings and uses a vector store for similarity search on these embeddings. This process allows GPTCache to identify and retrieve similar or related queries from the cache storage.

<a target="_blank" href="https://github.com/zilliztech/GPTCache/blob/main/docs/GPTCacheStructure.png">
<p align="center">
  <img src="https://github.com/zilliztech/GPTCache/blob/main/docs/GPTCacheStructure.png" alt="Cache Structure" width=600 height=200>
</p>
</a>

# Installation
To use the caching plugin functionality, you need to install the `gptcache` library first. You can do this using pip:

```bash
pip install -r requirements.txt
```

# Usage
## Initializing

Before using the functionality of caching plugin, you need to initialize the caching plugin with the desired configuration. The following code demonstrates how to initialize caching plugin:

```python
from intel_extension_for_transformers.neural_chat.pipeline.plugins.cache import CachePlugin
cache_plugin = CachePlugin()
cache_plugin.init_similar_cache_from_config()
```

## Caching Data

Once cache plugin is initialized, you can start caching data using the `put`` function. Here's an example of how to cache data:

```python
prompt = "Tell me about Intel Xeon Scable Processors."
response = chatbot.predict(prompt)
cache_plugin.put(prompt, response)
```

## Retrieving Cached Data

To retrieve cached data, use the get function. Provide the same prompt/question text used for caching, and it will return the cached answer. Here's an example:

```python
answer = cache_plugin.get("Tell me about Intel Xeon Scable Processors.")
```
