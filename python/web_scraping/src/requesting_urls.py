from typing import Dict, Optional
import time
import requests
import requests_cache

# html cache
requests_cache.install_cache(
    "cache",
    expire_after=3600
)

def get_html(url: str, params: Optional[Dict] = None, output: Optional[str] = None) -> str:
    """Gets an HTML page and return its contents.

    Arguments:
        url (str) : The URL to retrieve.
        params (dict, optional) : URL parameters to add.
        output (str, optional) : (optional) path where output should be saved.
    Returns:
        html (str) : The HTML of the page, as text.
    """

    headers = {
        "User-Agent": "BorgeWebScraper/1.0 (educational project)"
    }

    # passing the optional parameters argument to the get function
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=10
    )

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        wait_time = int(retry_after) if retry_after else 5

        print(f"Rate limited. Waiting {wait_time} seconds...")
        time.sleep(wait_time)

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )

    response.raise_for_status()

    html_str = response.text

    if output:
        # if output is specified, the response txt and url get printed to a
        # txt file with the name in `output`
        with open(output, 'w') as file:
            file.write(url+"\n")
            file.write(html_str)

    return html_str
