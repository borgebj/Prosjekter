import re
from typing import Tuple

## -- Task 3 (IN3110 optional, IN4110 required) -- ##

# create array with all names of months
month_names = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def get_date_patterns() -> Tuple[str, str, str]:
    """Return strings containing regex pattern for ye ar, month, day

    Arguments:
        None
    Returns:
        year, month, day (tuple) : Tuple containing regular expression patterns for each field
    """

    # Regex to capture days, months and years with numbers

    # year matches a 4-digit number between at least 1000-2029
    # \b[1-9]\d{3}\b
    year = r'''
    \b      # boundary end
    [1-9]   # digit 1-9
    \d{3}   # any 3 digit
    \b      # boundary end
    '''

    # month matches month names or month numbers
    # \b(?:{jan}|{feb}|{mar}|{apr}|{may}|{jun}|{jul}|{aug}|{sep}|{oct}|{nov}|{dec})|{iso_month_format}\b
    jan = r'\bjan(?:uary)?\b'
    feb = r'\bfeb(?:ruary)?\b'
    mar = r'\bmar(?:ch)?\b'
    apr = r'\bapr(?:il)?\b'
    may = r'\bmay\b'
    jun = r'\bjun(?:e)?\b'
    jul = r'\bjul(?:y)?\b'
    aug = r'\baug(?:ust)?\b'
    sep = r'\bsep(?:tember)?\b'
    oct = r'\boct(?:ober)?\b'
    nov = r'\bnov(?:ember)?\b'
    dec = r'\bdec(?:ember)?\b'
    iso_month_format = r'(?:0\d|1[0-2])'

    month = rf'''
    \b                                      # start boundary
    (?:                                     # start non-capturing
    {jan}|{feb}|{mar}|{apr}|{may}|{jun}     # named format 
    |{jul}|{aug}|{sep}|{oct}|{nov}|{dec}
    )                                       # end non-capturing
    |                                       # or
    {iso_month_format}                      # iso format
    \b                                      # end boundary
    '''

    # day matches a number, which may or may not be zero-padded
    # \b\d?\d\b
    day = r'''
    \b      # boundary start
    \d?     # optional digit
    \d      # any digit
    \b      # boundary end
    '''

    return year, month, day


def convert_month(s: str) -> str:
    """Converts a string month to number (e.g. 'September' -> '09'.

    Arguments:
        month_name (str) : month name
    Returns:
        month_number (str) : month number as zero-padded string
    """
    # If already digit do nothing
    if s.isdigit():
        return s
    # Else use help-function and list to make padded number
    else:
        index = month_names.index(s.capitalize())
        return zero_pad((str(index + 1)))


def zero_pad(n: str) -> str:
    """zero-pad a number string

    turns '2' into '02'

    Arguments:
        number (string): a number, either 1 or 2-digit
    Returns:
        number (string) : same number but padded with zero in-front

    """
    return n.zfill(2)


def find_dates(text: str, output: str = None) -> list:
    """Finds all dates in a text using reg ex

    Arguments:
        text (string) : A string containing html text from a website
        output (string) : Name of the output-file to be created
    Returns:
        results (list) : A list with all the dates found
    """
    year, month, day = get_date_patterns()

    # Date on format YYYY/MM/DD - ISO
    ISO = rf'({year})-({month})-({day})'

    # Date on format DD/MM/YYYY
    DMY = rf'({day})\s({month})\s({year})'

    # Date on format MM/DD/YYYY
    MDY = rf'({month})\s({day}),\s({year})'

    # Date on format YYYY/MM/DD
    YMD = rf'({year})\s({month})\s({day})'

    # list with all supported formats
    formats = [ISO, DMY, MDY, YMD]
    dates = []

    # find all dates in any format in text+
    for format in formats:

        # finds all matches using custom regex
        match = re.findall(format, text, flags=re.IGNORECASE|re.VERBOSE)
        for tuple in match:

            # supports ISO and YMD
            t_year = tuple[0]
            t_month = tuple[1]
            t_day = tuple[2]

            # check for DMY and MDY - swap accordingly
            if len(tuple[2]) > 2:
                t_year, t_day = t_day, t_year
                if len(tuple[0]) > 2:
                    t_day, t_month = t_month, t_day

            # convert to Y/M/D
            y_m_d = t_year + "/" + convert_month(t_month) + "/" + zero_pad(t_day)
            # adds to dates
            dates.append(y_m_d)

    # Write to file if wanted
    if output:
        with open(output, 'w') as file:
            file.write("\n".join(dates))

    return dates
