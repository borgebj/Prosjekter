# IN4110-borgebj


## How to run script
- Files such as **time_planner.py** can be run. This file has pre-defined input and uses example-dates from past few years
  - can be run using `python3 time_planner.py`

### Testing
- Testing can be done individually via pytest together with the various files
  - `pytest -v tests/test_requesting_urls.py` - Task 1
  - `pytest -v tests/test_filter_urls.py` - Task 2-3
  - `pytest -v tests/test_collect_dates.py` - Task 4
  - `pytest -v tests/test_time_planner.py` - Task 5-7
  - `pytest -v test/fetch_player_statistics.py` - Task 9-10
- Testing everything at once can be done using one command:
  - `pytest -vv tests`


## Required dependencies and packages
  Supports **python** version 3.5+

- **pytest** used for testing the files
- **Requests** used as library for html requests
- **BeautifulSoup4** used for HTML parsing
- **Pandas** for data analysis and table creation
- **Matplotlib** for plotting graphs
- **Tabulate** for printing tabular data. Table to markdown
- **requests-cache** for performance improvement of player statistics _(optional)_

## How to install dependencies and packages

|   _Package_    |                commando                 | 
|:--------------:|:---------------------------------------:|
|     Pytest     |          ` pip install pytest`          |
|    Requests    |    `python3 -m pip install requests`    |
| BeautifulSoup  | `python3 -m pip install beautifulsoup4` |
|     Pandas     |     `python3 -m pip install pandas`     |
|    Tabulate    |    `python3 -m pip install tabulate`    |
|   Matplotlib   |   `python3 -m pip install matplotlib`   |
 | requests-cache | `python3 -m pip install requests-cache` |

- To check if these packages are installed, pip can be used again:
  - `pip show [package_name]`