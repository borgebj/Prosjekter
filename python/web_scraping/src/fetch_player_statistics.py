import re
import os
from operator import itemgetter
from typing import Dict, List
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from requesting_urls import get_html
from filter_urls import finish_url
import pandas as pd


base_url = "https://en.wikipedia.org"


def find_best_players(url: str) -> None:
    """Find the best players in the semifinals of the nba.

    This is the top 3 scorers from every team in semifinals.
    Displays plot over points, assists, rebounds

    Arguments:
        html (str) : html string from wiki basketball
    Returns:
        None
    """
    # gets the teams, make sure only 8 teams
    teams = get_teams(url)
    assert len(teams) == 8, "more than 8 teams"

    # gets the player for every team and stores in dict,
    # using (get_players)
    all_players = {}
    for team in teams:
        team_name = team["name"]
        team_url = team["url"]
        all_players[team_name] = get_players(team_url)

    # gets player statistics for each player,
    # using get_player_stats
    for team, players in all_players.items():
        new_player_info = []
        for player in players:

            # gets player name and stats of player - merges old and stats to new info list
            stats = get_player_stats(player["url"], team)
            new_stats = {**player, **stats}
            new_player_info.append(new_stats)

            # in case of errors
            assert len(player) != 0, f"\nerror in player-data for {player['name'], player['url']}\n"
            assert len(stats) != 0, f"\nerror in stat-retrieval {player['name'], player['url']}\n"
            assert len(new_stats) != 0, f"\nerror in new stat creation {player['name'], player['url']}\n"

        all_players[team] = new_player_info


    # at this point, we should have a dict (all_players) of the form:
    # {
    #     "team name": [
    #         {
    #             "name": "player name",
    #             "url": "https://player_url",
    #             # added by get_player_stats
    #             "points": 5,
    #             "assists": 1.2,
    #             # ...,
    #         },
    #     ]
    # }

    # Select top 3 for each team by points:
    best = {}
    top_stat = "points"
    for team, players in all_players.items():

        # sorts the player using lambda - highest points first
        sorted_players = sorted(players, key=itemgetter(top_stat), reverse=True)

        # Sort and extract top 3 based on points
        top_3 = [sorted_players[0], sorted_players[1], sorted_players[2]]
        best[team] = top_3

    stats_to_plot = ["points", "assists", "rebounds"]
    for stat in stats_to_plot:
        plot_best(best, stat=stat)


def plot_best(best: Dict[str, List[Dict]], stat: str = "points") -> None:
    """Plots a single stat for the top 3 players from every team.

    Creates DataFrame table from data which is then used to plot chart

    DataFrame layout
            team	p1		p2		p3
        0	t_name	point	point	point
        1	t_name	point	point	point
        2	t_name	point	point	point
        3	t_name	point	point	point
        4	t_name	point	point	point
        5	t_name	point	point	point
        6	t_name 	point	point	point
        7	t_name	point	point	point
        8	t_name	point	point	point

    Arguments:
        best (dict) : dict with the top 3 players from every team
            has the form:

            best = {
                "team name": [
                    {
                        "name": "player name",
                        "points": 5,
                        ...
                    },
                ],
            }

            where the _keys_ are the team name,
            and the _values_ are lists of length 3,
            containing dictionaries about each player,
            with their name and stats.

        stat (str) : [points | assists | rebounds] which stat to plot.
            Should be a key in the player info dictionary.
    Returns:
        None/
    """
    first = {stat: [], "name": []}
    second = {stat: [], "name": []}
    third = {stat: [], "name": []}

    all_teams = []
    all_names = [first["name"], second["name"], third["name"]]

    # iterates through data to map it into new lists
    for team, players in best.items():

        # lists all teams together (8)
        all_teams.append(team)

        # gets stats into dictionary
        first[stat].append(players[0][stat])
        second[stat].append(players[1][stat])
        third[stat].append(players[2][stat])

        # gets names into dictionary
        first["name"].append(players[0]["name"])
        second["name"].append(players[1]["name"])
        third["name"].append(players[2]["name"])

    # creates a Pandas DataFrame using new lists containing data from best-dict
    df = pd.DataFrame({"team": all_teams, "best": first[stat], "second": second[stat], "third": third[stat]})
    color_dict = {"best": "#e24a33", "second": "#348abd", "third": "#988ed5"}

    # plots the Pandas DataFrame
    ax = df.plot(kind="bar",
                 zorder=2,
                 color=[color_dict.get(x, "#333333") for x in df.columns[1:]],
                 label=all_teams,
                 linewidth=0.5,
                 edgecolor="white")

    # labeling
    plt.title(stat.capitalize() + " for top 3 players in all teams", fontsize=8)
    plt.xticks(range(len(all_teams)), labels=all_teams, rotation=60, size=8)
    plt.xlabel("Teams", size=9)
    plt.ylabel(stat.capitalize(), size=9)

    # sets custom bar-labels to player-names
    for i in range(len(ax.containers)):
        ax.bar_label(ax.containers[i],
                     labels=all_names[i],
                     padding=3,
                     rotation=90,
                     fontsize=5)

    # layout-manipulation
    ax.set_facecolor("#e5e5e5")
    plt.tight_layout()
    x, y = plt.ylim()
    plt.ylim(x, y*1.9)
    plt.grid(True, zorder=0, color="#f8f8f8")
    plt.legend(["Best", "Second", "Third"],
               title="Player ranking within team based on "+stat,
               loc="upper right",
               facecolor="#e5e5e5",
               prop={'size': 5},
               fontsize=4,
               title_fontsize=5)

    stats_dir = "NBA_Player_statistics"
    os.makedirs(stats_dir, exist_ok=True)
    filename = stat+".png"
    print(f"Creating {filename}")
    plt.savefig(f"{stats_dir}/{filename}", dpi=300)


def get_teams(url: str) -> list:
    """Extracts all the teams that were in the semi-finals in nba

    Arguments:
        - url (str) : url of the nba finals wikipedia page
    Returns:
        teams (list) : list with all teams
            Each team is a dictionary of {'name': team name, 'url': team page
    """
    # Get the table
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find(id="Bracket").find_next("table")

    # find all rows in table
    rows = table.find_all("tr")
    rows = rows[2:]
    # maybe useful: identify cells that look like 'E1' or 'W5', etc.
    seed_pattern = re.compile(r"^[EW][1-8]$")

    # builds a set of team names in the semifinal
    # and a dict of {team name: team url}

    team_links = {}  # dict of team name: team url
    in_semifinal = set()  # set of teams in the semifinal

    # Loop over every row and extract teams from semi finals
    # also locate the links tot he team pages from the First Round column
    for row in rows:
        cols = row.find_all("td")
        # useful for showing structure
        # print([c.get_text(strip=True) for c in cols])

        # TODO:
        # 1. if First Round column, record team link from `a` tag
        # 2. if semifinal column, record team name

        # quarterfinal, E1/W8 is in column 1
        # team name, link is in column 2
        if len(cols) >= 3 and seed_pattern.match(cols[1].get_text(strip=True)):
            team_col = cols[2]
            a = team_col.find("a")
            team_links[team_col.get_text(strip=True)] = urljoin(base_url, a["href"])

        elif len(cols) >= 4 and seed_pattern.match(cols[2].get_text(strip=True)):
            team_col = cols[3]
            in_semifinal.add(team_col.get_text(strip=True))

        elif len(cols) >= 5 and seed_pattern.match(cols[3].get_text(strip=True)):
            team_col = cols[4]
            in_semifinal.add(team_col.get_text(strip=True))

    # return list of dicts (there will be 8):
    # [
    #     {
    #         "name": "team name",
    #         "url": "https://team url",
    #     }
    # ]

    assert len(in_semifinal) == 8
    return [
        {
            "name": team_name.rstrip("*"),
            "url": team_links[team_name],
        }
        for team_name in in_semifinal
    ]


def get_players(team_url: str) -> list:
    """Gets all the players from a team that were in the roster for semi finals

    Arguments:
        team_url (str) : the url for the team
    Returns:
        player_infos (list) : list of player info dictionaries
            with form: {'name': player name, 'url': player wikipedia page url}
    """
    print(f"Finding players in {team_url}")

    # Gets the table
    html = get_html(team_url)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find(id="Roster").find_next("table", {"class": "sortable"})

    players = []
    # Loop over every row and gets the names from the roster
    rows = table.find_all("tr")[1:]
    for row in rows:
        name_dict = {}

        # gets data from each row
        cells = row.find_all("td")

        # extracts name-row and their anchor-tags containing links
        name_tag = cells[2]
        anchor_tag = name_tag.find("a")

        # labels name and anchor from extraction
        name = cells[2].text.strip().replace(u'\xa0', u' ')
        short_link = anchor_tag.get("href")
        full_link = finish_url(short_link, base_url)

        # adds player to dict and their url
        name_dict["name"] = name
        name_dict["url"] = full_link

        players.append(name_dict)

    # returns list of players and their url
    return players


def get_player_stats(player_url: str, team: str) -> dict:
    """Gets the player stats for a player in a given team

    Arguments:
        player_url (str) : url for the wiki page of player
        team (str) : the name of the team the player plays for
    Returns
        stats (dict) : dictionary with the keys (at least): points, assists, and rebounds keys
    """
    print(f"Fetching stats for player in {player_url}")

    # Gets the table with stats
    html = get_html(player_url)
    soup = BeautifulSoup(html, "html.parser")
    section = soup.find(id="Regular_season")

    stats = {"rebounds": 0.0, "assists": 0.0, "points": 0.0}
    desired_year = "2021–22"

    # checks if regular season table exists
    if not section:
        return stats

    table = section.find_next("table", {"class": "sortable"})

    # Loops over rows and extract the stats
    rows = table.find_all("tr")
    rows = rows[1:]
    for row in rows:
        cells = row.find_all("td")
        year = cells[0].text.strip()
        team_name = cells[1].text.strip()

        # checks if wanted team in row with correct year
        if team in team_name and desired_year in year:
            # Check correct team (some players change team within season)
            rpg = float(cells[8].text.replace("*", ""))
            apg = float(cells[9].text.replace("*", ""))
            ppg = float(cells[12].text.replace("*", ""))

            # load stats from columns and adds them to dictionary
            stats["rebounds"] = rpg
            stats["assists"] = apg
            stats["points"] = ppg

    # rounds the floats and returns
    return {k: round(v, 1) for (k, v) in stats.items()}


# run the whole thing if called as a script, for quick testing
if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/2022_NBA_playoffs"
    find_best_players(url)
