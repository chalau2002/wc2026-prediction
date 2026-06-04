const roundOrder = [
  "Round of 32",
  "Round of 16",
  "Quarter-final",
  "Semi-final",
  "Third-place playoff",
  "Final",
];

const roundLabels = {
  "Round of 32": "Round of 32",
  "Round of 16": "Round of 16",
  "Quarter-final": "Quarter-finals",
  "Semi-final": "Semi-finals",
  "Third-place playoff": "Third-place match",
  "Final": "Final",
};

let dashboardData;
let rankingSort = { key: "p_champion", direction: "desc" };
let selectedTeamName = null;

const teamFlags = {
  Algeria: "🇩🇿",
  Argentina: "🇦🇷",
  Australia: "🇦🇺",
  Austria: "🇦🇹",
  Belgium: "🇧🇪",
  "Bosnia and Herzegovina": "🇧🇦",
  Brazil: "🇧🇷",
  Canada: "🇨🇦",
  "Cape Verde": "🇨🇻",
  Colombia: "🇨🇴",
  Croatia: "🇭🇷",
  Curaçao: "🇨🇼",
  "Czech Republic": "🇨🇿",
  "DR Congo": "🇨🇩",
  Ecuador: "🇪🇨",
  Egypt: "🇪🇬",
  England: "\u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F}",
  France: "🇫🇷",
  Germany: "🇩🇪",
  Ghana: "🇬🇭",
  Haiti: "🇭🇹",
  Iran: "🇮🇷",
  Iraq: "🇮🇶",
  "Ivory Coast": "🇨🇮",
  Japan: "🇯🇵",
  Jordan: "🇯🇴",
  Mexico: "🇲🇽",
  Morocco: "🇲🇦",
  Netherlands: "🇳🇱",
  "New Zealand": "🇳🇿",
  Norway: "🇳🇴",
  Panama: "🇵🇦",
  Paraguay: "🇵🇾",
  Portugal: "🇵🇹",
  Qatar: "🇶🇦",
  "Saudi Arabia": "🇸🇦",
  Scotland: "\u{1F3F4}\u{E0067}\u{E0062}\u{E0073}\u{E0063}\u{E0074}\u{E007F}",
  Senegal: "🇸🇳",
  "South Africa": "🇿🇦",
  "South Korea": "🇰🇷",
  Spain: "🇪🇸",
  Sweden: "🇸🇪",
  Switzerland: "🇨🇭",
  Tunisia: "🇹🇳",
  Turkey: "🇹🇷",
  "United States": "🇺🇸",
  Uruguay: "🇺🇾",
  Uzbekistan: "🇺🇿",
};

function pct(value) {
  return `${Math.round(Number(value) * 1000) / 10}%`;
}

function pct2(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function byId(id) {
  return document.getElementById(id);
}

function defaultTeamName(data) {
  return data.teams.some((team) => team.team === "Portugal") ? "Portugal" : data.teams[0].team;
}

function teamLabel(team) {
  return `<span class="team-name"><span class="flag" aria-hidden="true">${teamFlags[team] || "🏳️"}</span><span>${team}</span></span>`;
}

function metricValue(team, key) {
  return team.metrics.find((metric) => metric.key === key)?.value || 0;
}

function topGroupValue(teamName) {
  for (const group of dashboardData.groups) {
    const row = group.positionProbabilities.find((item) => item.team === teamName);
    if (row) return row.positions["1"] || 0;
  }
  return 0;
}

function rankingSortValue(team, key) {
  if (key === "team") return team.team;
  if (key === "rank") return team.rankFinal;
  if (key === "top_group") return topGroupValue(team.team);
  return metricValue(team, key);
}

function renderRanking() {
  const sortedTeams = dashboardData.teams
    .slice()
    .sort((a, b) => {
      const aValue = rankingSortValue(a, rankingSort.key);
      const bValue = rankingSortValue(b, rankingSort.key);
      const direction = rankingSort.direction === "asc" ? 1 : -1;

      if (typeof aValue === "string") {
        return aValue.localeCompare(bValue) * direction;
      }
      return (aValue - bValue) * direction;
    });

  byId("rankingBody").innerHTML = sortedTeams
    .map(
      (team, index) => `
        <tr>
          <td class="position">${index + 1}</td>
          <td class="team-cell">${teamLabel(team.team)}</td>
          <td>${pct2(topGroupValue(team.team))}</td>
          <td>${pct2(metricValue(team, "p_round_of_32"))}</td>
          <td>${pct2(metricValue(team, "p_round_of_16"))}</td>
          <td>${pct2(metricValue(team, "p_quarter_final"))}</td>
          <td>${pct2(metricValue(team, "p_semi_final"))}</td>
          <td>${pct2(metricValue(team, "p_final"))}</td>
          <td>${pct2(metricValue(team, "p_runner_up"))}</td>
          <td>${pct2(metricValue(team, "p_third_place"))}</td>
          <td class="strong-stat">${pct2(metricValue(team, "p_champion"))}</td>
        </tr>
      `,
    )
    .join("");
}

function setupRankingSort() {
  document.querySelectorAll(".sort-button").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sort;
      const sameColumn = rankingSort.key === key;
      rankingSort = {
        key,
        direction: sameColumn && rankingSort.direction === "desc" ? "asc" : "desc",
      };
      document.querySelectorAll(".sort-button").forEach((item) => {
        item.classList.toggle("active", item.dataset.sort === key);
        item.dataset.direction = item.dataset.sort === key ? rankingSort.direction : "";
      });
      renderRanking();
    });
  });
}

function renderTeam(teamName) {
  const team = dashboardData.teams.find((item) => item.team === teamName) || dashboardData.teams[0];
  byId("selectedTeam").innerHTML = teamLabel(team.team);
  byId("selectedGroup").textContent = team.group;

  const championMetric = team.metrics.find((metric) => metric.key === "p_champion");
  byId("championValue").textContent = pct2(championMetric?.value || 0);

  byId("teamMetrics").innerHTML = team.metrics
    .map(
      (metric) => `
        <article class="metric">
          <div class="metric-top">
            <span class="metric-label">${metric.label}</span>
            <strong class="metric-value">${pct2(metric.value)}</strong>
          </div>
          <div class="bar" aria-hidden="true">
            <div class="bar-fill" style="width: ${Number(metric.value) * 100}%"></div>
          </div>
        </article>
      `,
    )
    .join("");

  renderTeamGames(team.team);
}

function teamMatches(teamName) {
  return dashboardData.groups
    .flatMap((group) => group.matches.map((match) => ({ ...match, group: group.group })))
    .filter((match) => match.home === teamName || match.away === teamName);
}

function matchProbabilityCard(match) {
  return `
    <article class="modal-match">
      <div class="modal-match-title">
        <span>${teamLabel(match.home)}</span>
        <strong>vs</strong>
        <span>${teamLabel(match.away)}</span>
      </div>
      <div class="match-bars">
        <div class="match-bar-row">
          <span>${match.home}</span>
          <strong>${pct(match.homeWin)}</strong>
          <div class="mini-bar"><span style="width: ${Number(match.homeWin) * 100}%"></span></div>
        </div>
        <div class="match-bar-row">
          <span>Draw</span>
          <strong>${pct(match.draw)}</strong>
          <div class="mini-bar"><span style="width: ${Number(match.draw) * 100}%"></span></div>
        </div>
        <div class="match-bar-row">
          <span>${match.away}</span>
          <strong>${pct(match.awayWin)}</strong>
          <div class="mini-bar"><span style="width: ${Number(match.awayWin) * 100}%"></span></div>
        </div>
      </div>
      <p class="score-note">Most likely score: <strong>${match.mostLikelyScore}</strong> · ${pct(match.scoreConfidence)}</p>
    </article>
  `;
}

function renderTeamGames(teamName) {
  const matches = teamMatches(teamName);
  byId("teamGames").innerHTML = matches.map(matchProbabilityCard).join("");
}

function renderGroups() {
  byId("groupsGrid").innerHTML = dashboardData.groups
    .map(
      (group) => `
        <article class="group-card">
          <div class="group-head">
            <h3>Group ${group.group}</h3>
            <button class="results-button" type="button" data-group="${group.group}">Results</button>
          </div>
          <table>
            <thead>
              <tr>
                <th>Team</th>
                <th>1st</th>
                <th>2nd</th>
                <th>3rd</th>
                <th>4th</th>
              </tr>
            </thead>
            <tbody>
              ${group.positionProbabilities
                .map(
                  (row, index) => `
                    <tr class="${index < 2 ? "qualified" : index === 2 ? "possible" : ""}">
                      <td class="team-cell">${teamLabel(row.team)}</td>
                      <td class="strong-stat">${pct(row.positions["1"])}</td>
                      <td>${pct(row.positions["2"])}</td>
                      <td>${pct(row.positions["3"])}</td>
                      <td>${pct(row.positions["4"])}</td>
                    </tr>
                  `,
                )
                .join("")}
            </tbody>
          </table>
        </article>
      `,
    )
    .join("");

  document.querySelectorAll(".results-button").forEach((button) => {
    button.addEventListener("click", () => openGroupModal(button.dataset.group));
  });
}

function openGroupModal(groupLetter) {
  const group = dashboardData.groups.find((item) => item.group === groupLetter);
  if (!group) return;

  byId("groupModalTitle").textContent = `Group ${group.group} match probabilities`;
  byId("groupModalBody").innerHTML = `
    <div class="modal-match-list">
      ${group.matches.map(matchProbabilityCard).join("")}
    </div>
  `;
  byId("groupModal").hidden = false;
}

function closeGroupModal() {
  byId("groupModal").hidden = true;
}

function setupGroupModal() {
  byId("closeGroupModal").addEventListener("click", closeGroupModal);
  byId("groupModal").addEventListener("click", (event) => {
    if (event.target.id === "groupModal") closeGroupModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !byId("groupModal").hidden) closeGroupModal();
  });
}

function renderBracket() {
  const matches = Object.fromEntries(dashboardData.bracket.map((match) => [match.matchId, match]));
  const left = {
    r32: [73, 75, 74, 77, 83, 84, 81, 82],
    r16: [89, 90, 93, 94],
    qf: [97, 98],
    sf: [101],
  };
  const right = {
    sf: [102],
    qf: [99, 100],
    r16: [91, 92, 95, 96],
    r32: [76, 78, 79, 80, 86, 88, 85, 87],
  };

  const card = (matchId, side = "") => {
    const match = matches[matchId];
    if (!match) return "";
    const homeIsFav = match.homePassProbability >= match.awayPassProbability;
    const awayIsFav = match.awayPassProbability > match.homePassProbability;

    return `
      <article class="bracket-card ${side}" data-match-id="${match.matchId}">
        <div class="bracket-team ${homeIsFav ? "favored" : ""}">
          ${teamLabel(match.home)}
          <strong>${pct(match.homePassProbability)}</strong>
        </div>
        <div class="bracket-team ${awayIsFav ? "favored" : ""}">
          ${teamLabel(match.away)}
          <strong>${pct(match.awayPassProbability)}</strong>
        </div>
      </article>
    `;
  };

  const column = (label, ids, side = "") => `
    <section class="bracket-column ${side}">
      <h3>${label}</h3>
      <div class="bracket-column-matches count-${ids.length}">
        ${ids.map((id) => card(id, side)).join("")}
      </div>
    </section>
  `;

  byId("bracketGrid").innerHTML = `
    <div class="bracket-scroll">
      <div class="bracket-board">
        <svg class="bracket-lines" aria-hidden="true"></svg>
        ${column("Round of 32", left.r32, "left")}
        ${column("Round of 16", left.r16, "left")}
        ${column("Quarter-finals", left.qf, "left")}
        ${column("Semi-final", left.sf, "left")}
        <section class="bracket-center">
          <div class="champion-block">
            <span>Most likely final</span>
            ${card(104, "center")}
          </div>
          <div class="third-place-block">
            <span>Third-place match</span>
            ${card(103, "center")}
          </div>
        </section>
        ${column("Semi-final", right.sf, "right")}
        ${column("Quarter-finals", right.qf, "right")}
        ${column("Round of 16", right.r16, "right")}
        ${column("Round of 32", right.r32, "right")}
      </div>
    </div>
  `;

  requestAnimationFrame(drawBracketLines);
}

function drawBracketLines() {
  const board = document.querySelector(".bracket-board");
  const svg = document.querySelector(".bracket-lines");
  if (!board || !svg || board.offsetWidth === 0 || board.offsetHeight === 0) return;

  const boardRect = board.getBoundingClientRect();
  const point = (matchId, edge) => {
    const card = board.querySelector(`[data-match-id="${matchId}"]`);
    if (!card) return null;
    const rect = card.getBoundingClientRect();
    const x = edge === "left" ? rect.left - boardRect.left : rect.right - boardRect.left;
    return {
      x,
      y: rect.top - boardRect.top + rect.height / 2,
    };
  };

  const pairPath = (sourceA, sourceB, target, sourceEdge, targetEdge) => {
    const a = point(sourceA, sourceEdge);
    const b = point(sourceB, sourceEdge);
    const t = point(target, targetEdge);
    if (!a || !b || !t) return "";

    const midX = sourceEdge === "right"
      ? Math.min(a.x, t.x) + Math.abs(t.x - a.x) / 2
      : Math.max(a.x, t.x) - Math.abs(t.x - a.x) / 2;
    const minY = Math.min(a.y, b.y, t.y);
    const maxY = Math.max(a.y, b.y, t.y);

    return `
      <path d="M ${a.x} ${a.y} H ${midX}" />
      <path d="M ${b.x} ${b.y} H ${midX}" />
      <path d="M ${midX} ${minY} V ${maxY}" />
      <path d="M ${midX} ${t.y} H ${t.x}" />
    `;
  };

  const singlePath = (source, target, sourceEdge, targetEdge) => {
    const a = point(source, sourceEdge);
    const t = point(target, targetEdge);
    if (!a || !t) return "";
    const midX = sourceEdge === "right"
      ? Math.min(a.x, t.x) + Math.abs(t.x - a.x) / 2
      : Math.max(a.x, t.x) - Math.abs(t.x - a.x) / 2;
    return `<path d="M ${a.x} ${a.y} H ${midX} V ${t.y} H ${t.x}" />`;
  };

  const lines = [
    pairPath(73, 75, 89, "right", "left"),
    pairPath(74, 77, 90, "right", "left"),
    pairPath(83, 84, 93, "right", "left"),
    pairPath(81, 82, 94, "right", "left"),
    pairPath(89, 90, 97, "right", "left"),
    pairPath(93, 94, 98, "right", "left"),
    pairPath(97, 98, 101, "right", "left"),
    singlePath(101, 104, "right", "left"),
    pairPath(76, 78, 91, "left", "right"),
    pairPath(79, 80, 92, "left", "right"),
    pairPath(86, 88, 95, "left", "right"),
    pairPath(85, 87, 96, "left", "right"),
    pairPath(91, 92, 99, "left", "right"),
    pairPath(95, 96, 100, "left", "right"),
    pairPath(99, 100, 102, "left", "right"),
    singlePath(102, 104, "left", "right"),
  ].join("");

  svg.setAttribute("viewBox", `0 0 ${board.offsetWidth} ${board.offsetHeight}`);
  svg.innerHTML = lines;
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
      document.querySelectorAll(".tab-view").forEach((view) => view.classList.remove("active"));
      button.classList.add("active");
      byId(`${button.dataset.tab}View`).classList.add("active");
      if (button.dataset.tab === "bracket") requestAnimationFrame(drawBracketLines);
    });
  });
}

function applyDashboardData(nextData) {
  dashboardData = nextData;
  const select = byId("teamSelect");
  select.innerHTML = dashboardData.teams
    .slice()
    .sort((a, b) => a.team.localeCompare(b.team))
    .map((team) => `<option value="${team.team}">${teamFlags[team.team] || "🏳️"} ${team.team}</option>`)
    .join("");
  const nextTeam = selectedTeamName && dashboardData.teams.some((team) => team.team === selectedTeamName)
    ? selectedTeamName
    : defaultTeamName(dashboardData);
  select.value = nextTeam;
  selectedTeamName = nextTeam;

  renderRanking();
  renderTeam(nextTeam);
  renderGroups();
  renderBracket();
}

async function loadDashboardData() {
  const response = await fetch(`dashboard-data.json?ts=${Date.now()}`, { cache: "no-store" });
  const nextData = await response.json();
  applyDashboardData(nextData);
}

async function init() {
  await loadDashboardData();

  const select = byId("teamSelect");
  select.addEventListener("change", () => {
    selectedTeamName = select.value;
    renderTeam(select.value);
  });

  setupRankingSort();
  setupTabs();
  setupGroupModal();
  window.addEventListener("resize", drawBracketLines);

  setInterval(() => {
    loadDashboardData().catch((error) => console.warn("Dashboard refresh failed:", error));
  }, 60000);
}

init().catch((error) => {
  document.body.innerHTML = `<main><section class="team-panel"><h1>Error loading data</h1><p>${error.message}</p></section></main>`;
});
