import re
from copy import copy
from dataclasses import dataclass

import bs4
import pandas as pd
from bs4 import BeautifulSoup
from requesting_urls import get_html

## --- Task 5, 6, and 7 ---- ##

# Dict over all types of events
event_types = {
    "DH": "Downhill",
    "SL": "Slalom",
    "GS": "Giant Slalom",
    "SG": "Super Giant slalom",
    "AC": "Alpine Combined",
    "PG": "Parallel Giant Slalom",
}


def time_plan(url: str) -> str:
    """Parses table from html text and extract desired information
    and saves in betting slip markdown file

    arguments:
        url (str) : URL for page with calendar table
    return:
        markdown (str) : string containing the markdown schedule
    """
    # Gets the page, parse html
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    # locates the table
    calendar = soup.find(id="Calendar")
    soup_table = calendar.find_next("table", {"class": "wikitable"})

    # extracts events into pandas data frame
    df = extract_events(soup_table)

    # Writes the schedule markdown and returns it
    return render_schedule(df)


@dataclass
class TableEntry:
    """Data class representing a single entry in a table
    Records text content, rowspan, and colspan attributes
    """

    text: str
    rowspan: int
    colspan: int


def extract_events(table: bs4.element.Tag) -> pd.DataFrame:
    """Gets the events from the table

    Arguments:
        table (bs4.element.Tag) : Table containing data
    Returns:
        df (DataFrame) : DataFrame containing filtered and parsed data
    """
    # Gets the table headers and saves their labels in `keys`
    headings = table.find_all("th")
    labels = [th.text.strip() for th in headings]
    data = []

    # Extracts the data in table, keeping track of colspan and rowspan
    rows = table.find_all("tr")
    rows = rows[1:]
    for tr in rows:
        cells = tr.find_all("td")
        row = []
        for cell in cells:
            colspan = 1
            rowspan = 1

            # checks if exist
            if "colspan" in cell.attrs:
                colspan = int(cell["colspan"])
            if "rowspan" in cell.attrs:
                rowspan = int(cell["rowspan"])

            text = cell.get_text().strip()
            row.append(
                TableEntry(
                    text=text,
                    rowspan=rowspan,
                    colspan=colspan,
                )
            )
        data.append(row)

    # at this point `data` should be a table (list of lists)
    # where each item is a TableEntry with row/colspan properties
    # expands TableEntries into a dense table
    all_data = expand_row_col_span(data)

    # List of desired columns - can add more
    wanted = ["Date", "Venue", "Type"]
    wanted = [key for key in wanted if key in labels]

    # Filters data into a dictionary of desired columns and their values
    filtered_data = filter_data(labels, all_data, wanted)

    # creates the actual pandas DataFrame with the filtered dictionary
    df = pd.DataFrame(filtered_data)

    return df


def render_schedule(data: pd.DataFrame) -> str:
    """Render the schedule data to markdown

    Arguments:
        data (DataFrame) : DataFrame containing table to write
    Returns:
        markdown (str) : the rendered schedule as markdown
    """

    def expand_event_type(type_key):
        """Expand event type key (SL) to full name (Slalom)
        Useful with pandas Series.apply
        """
        return event_types.get(type_key[:2], type_key)

    # uses apply and function over to apply full name to types
    data["Type"] = data["Type"].apply(expand_event_type)

    return data.to_markdown()


def strip_text(text: str) -> str:
    """Gets rid of cruft from table cells, footnotes and setting limit to 20 chars
    It is not required to use this function,
    but it may be useful.

    Arguments:
        text (str) : string to fix
    Returns:
        text (str) : the string fixed
    """
    return re.sub(r"\[[^\]]*\]", "", text).strip()


def filter_data(keys: list, data: list, wanted: list):
    """Filters away the columns not specified in wanted argument
    It is not required to use this function,
    but it may be useful.

    Arguments:
        - keys (list of strings) : list of all column names
        - data (list of lists) : data with rows and columns
        - wanted (list of strings) : list of wanted columns
    Returns:
        filtered_data (dictionary of lists) : the filtered data
            The subset of data in 'data',
            after discarding the columns not in 'wanted',
            turned into a dictionary for DataFrame-creation.
    """

    filtered_dict = {}
    for key in keys:
        if key in wanted:
            new_row = []
            index = keys.index(key)
            for lst in data:
                new_row.append(lst[index])
            filtered_dict[key] = new_row
    return filtered_dict


def expand_row_col_span(data):
    """Applies row/colspan to tabular data
    It is not required to use this function,
    but it may be useful.
    - Copies cells with colspan to columns to the right
    - Copies cells with rowspan to rows below
    - Returns raw data (removing TableEntry wrapper)

    Arguments:
        data_table (list) : data with rows and cols
            Table of the form:
            [
                [ # row
                    TableEntry(text='text', rowspan=2, colspan=1),
                ]
            ]
    Returns:
        new_data_table (list) : list of lists of strings
            [
                [
                    "text",
                    "text",
                    ...
                ]
            ]
            This should be a dense matrix (list of lists) of data,
            where all rows have the same length,
            and all values are `str`.
    """

    # first, apply colspan by duplicating across the column(s)
    new_data = []
    for row in data:
        new_row = []
        new_data.append(new_row)
        for entry in row:
            for _ in range(entry.colspan):
                new_entry = copy(entry)
                new_entry.colspan = 1
                new_row.append(new_entry)

    # apply row span by inserting copies in subsequent rows
    # in the same column
    for row_idx, row in enumerate(new_data):
        for col_idx, entry in enumerate(row):
            for offset in range(1, entry.rowspan):
                # copy to row(s) below
                target_row = new_data[row_idx + offset]
                new_entry = copy(entry)
                new_entry.rowspan = 1
                target_row.insert(col_idx, new_entry)
            entry.rowspan = 1

    # now that we've applied col/row span,
    # replace the table with the raw entries,
    # instead of the TableEntry objects
    return [[strip_text(entry.text) for entry in row] for row in new_data]


if __name__ == "__main__":
    # tests the script on the past few years by running it:
    for year in range(20, 23):
        url = (
            f"https://en.wikipedia.org/wiki/20{year}–{year+1}_FIS_Alpine_Ski_World_Cup"
        )
        print("\n", url)
        md = time_plan(url)
        print(md)
