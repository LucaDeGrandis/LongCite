import logging
import requests
from multiprocessing import Pool
from typing import List, Optional

logger = logging.getLogger(__name__)


def embed_with_retry(input):
    pack, api_key = input
    for _ in range(5):
        try:
            url, text_list = pack
            response = requests.post(
                url,
                json={
                    'model': 'embedding-2',
                    'input': text_list
                },
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                },
            )
            # print(response.text)
            response = response.json()
            data_list = []
            for i in range(len(text_list)):
                data_list.append({
                    'embedding': response['data'][i]['embedding']
                })
            response = {
                'data': data_list
            }
            return response
        except Exception as e:
            print(response.text)
            logging.warning('embed_with_retry exception: %s' % e)


class ZhipuEmbeddings:
    def __init__(
        self,
        url: Optional[str] = None,
        embedding_proc: int = 8,
        embedding_batch_size: int = 8,
        api_key: str = None,
    ):
        self.url, self.embedding_proc, self.embedding_batch_size = url, embedding_proc, embedding_batch_size
        self.api_key = api_key

    def _get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        batched_embeddings: List[List[float]] = []

        data_processed = []
        for i in range(0, len(texts), self.embedding_batch_size):
            text_list = texts[i: i + self.embedding_batch_size]
            data_processed.append((self.url, text_list))
        with Pool(self.embedding_proc) as p:
            result = list(p.imap(embed_with_retry, zip(data_processed, [self.api_key] * len(data_processed))))
        for response in result:
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        return batched_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_len_safe_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
