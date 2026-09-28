import re
from urllib.parse import urljoin

## -- Task 2 -- ##


# check start of url and finishes it
def finish_url(url: str, base_url: str):
    """Modifies a shortened url with its base_url,
    or completes it with https in-front

    Arguments:
        url (str) : url to modify/finish
        base_url (str) : url to add on to given url
    Returns:
        url (str) : modified/finished url
    """
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = base_url + url

    return url


def find_urls(
    html: str,
    base_url: str = "https://en.wikipedia.org",
    output: str = None,
) -> set:
    """Find all the url links in a html text using regex
        Writes urls to file if specified

    Arguments:
        html (str) : html string to parse
        base_url (str) : base-url to add to unfinished links
        output: (str) : Name of output-file
    Returns:
        urls (set) : set with all the urls found in html text
    """
    # create and compile regular expression(s)

    urls = set()
    # 1. finds all the anchor tags
    anchor_pat = re.compile(r'<a[^>]+>', flags=re.IGNORECASE)

    # 2. finds the urls after the href attribute between quotes
    href_pat = re.compile(r'href="([^"]+)"', flags=re.IGNORECASE)

    for anchor in anchor_pat.findall(html):
        # then, find the src attribute of the img, if any
        match = href_pat.search(anchor)
        if match:
            url = match.group(1).split("#")[0]

            # calls function to finish url
            url = finish_url(url, base_url)

            urls.add(url)

    # Write to file if requested
    if output:
        print(f"Writing to: {output}")
        with open(output, 'w') as file:
            file.write("\n".join(urls))

    # removes occurrences of null from splitting
    return set(filter(None, urls))


def find_articles(html: str, output=None, all_lan=True) -> set:
    """Finds all the wiki articles inside a html text. Make call to find urls, and filter

    Arguments:
        text (str) : the html text to parse
        output (Str) : Name of output-file
        all_lan (bool) : added for wiki race. Allows filtering of english-only articles if false
    Returns:
        urls (set) : a set with urls to all the articles found
    """
    urls = find_urls(html, output=output)

    # regex that matches words including wikipedia.org without semicolon
    # .*wikipedia\.org((?!\:|\;).)*$    -    for all languages
    if all_lan:
        pattern = r'''
        .*                  # starting with anything
        wikipedia\.org      # followed by wikipedia.org
        (                   # open group
        (?!\:|\;)           # checks following letter - not : or ;
        .                   # any character
        )                   # end group
        *$                  # ending in 0 or more occurrences of group
        '''
    # else: slight change in regex - added '.en' for only english articles
    else: pattern = r'.*en.wikipedia\.org((?!\:|\;).)*$'

    articles = set()
    article_pat = re.compile(pattern, flags=re.VERBOSE)

    for url in urls:
        if article_pat.search(url):
            articles.add(url)

    # Write to file if wanted
    if output:
        print(f"Writing to: {output}")
        with open(output, 'w') as file:
            file.write("\n".join(urls))

    return articles


## Regex example
def find_img_src(html: str):
    """Find all src attributes of img tags in an HTML string

    The returned set contains every found src attibute of an img tag in the given HTML.

    Arguments:
        html (str) : A string containing some HTML.
    Returns:
        src_set (set) : A set of strings containing image URLs
    """
    # img_pat finds all the <img alt="..." src="..."> snippets
    # this finds <img and collects everything up to the closing '>'
    img_pat = re.compile(r"<img[^>]+>", flags=re.IGNORECASE)
    # src finds the text between quotes of the `src` attribute
    src_pat = re.compile(r'src="([^"]+)"', flags=re.IGNORECASE)
    src_set = set()
    # first, find all the img tags
    for img_tag in img_pat.findall(html):
        # then, find the src attribute of the img, if any
        match = src_pat.search(img_tag)
        if match:
            src_set.add(match.group(1))
    return src_set
