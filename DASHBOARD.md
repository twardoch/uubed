# uubed Project Dashboard

[![Project Health](https://img.shields.io/badge/project%20health-monitoring-blue)](https://twardoch.github.io/uubed/)

## Overview

This dashboard shows the current health and activity of all uubed repositories. It updates automatically every 6 hours with fresh data from GitHub.

## 🔗 Quick Links

- **[Live Dashboard](https://twardoch.github.io/uubed/)** – Interactive project status
- **[Main Repository](https://github.com/twardoch/uubed)** – Project coordination and docs
- **[Rust Implementation](https://github.com/twardoch/uubed-rs)** – Core engine
- **[Python Package](https://github.com/twardoch/uubed-py)** – Python bindings
- **[Documentation](https://github.com/twardoch/uubed-docs)** – Guides and reference

## 📊 What's Tracked

### Repository Health
- **Activity Status** – Recent commits
- **CI/CD Status** – Build and test results
- **Release Status** – Latest versions
- **Issue Management** – Open issues and PRs

### Development Metrics
- **Commit Activity** – Commits in the last 30 days
- **Community Engagement** – Stars, forks, contributors
- **Code Quality** – Test coverage, build success rate
- **Release Cadence** – Release frequency

### Project Coordination
- **Cross-Repo Sync** – Version alignment, dependency updates
- **Documentation Coverage** – Completeness of API docs and guides
- **Performance Tracking** – Benchmarks and regressions

## 🎯 Health Indicators

Status colors indicate repository health:

- **🟢 Green** – Active, stable, up-to-date
- **🟡 Yellow** – Some issues (e.g., stale branches, minor failures)
- **🔴 Red** – Requires attention (broken builds, outdated releases, etc.)

## 📈 Key Performance Indicators

### Project-Wide KPIs
- **Overall Health Score** – % of healthy repositories
- **Development Velocity** – Total commits across repos
- **Community Growth** – Combined stars and forks
- **Release Frequency** – Releases in the past quarter

### Repository-Specific KPIs
- **Code Freshness** – Days since last commit
- **CI Reliability** – % of successful builds
- **Issue Resolution Time** – Avg. time to close issues
- **Documentation Coverage** – % of documented API

## 🔄 Update Schedule

- **Every 6 hours** – Full metrics refresh
- **On push to main** – Immediate health check
- **Manual trigger** – Via GitHub Actions workflow dispatch

## 📋 Repository Status Summary

| Repository     | Purpose              | Language  | Status                              |
|----------------|----------------------|-----------|-------------------------------------|
| [uubed](https://github.com/twardoch/uubed)       | Coordination         | Markdown  | ![Status](https://img.shields.io/badge/status-active-brightgreen) |
| [uubed-rs](https://github.com/twardoch/uubed-rs)    | Core engine          | Rust      | ![Status](https://img.shields.io/badge/status-active-brightgreen) |
| [uubed-py](https://github.com/twardoch/uubed-py)    | Python bindings      | Python    | ![Status](https://img.shields.io/badge/status-active-brightgreen) |
| [uubed-docs](https://github.com/twardoch/uubed-docs)  | Documentation        | Markdown  | ![Status](https://img.shields.io/badge/status-active-brightgreen) |

## 🛠️ Dashboard Features

### Interactivity
- **Repository Cards** – Click through to GitHub
- **Workflow Links** – View CI/CD results directly
- **Issue Tracking** – See open issues and PRs at a glance
- **Release Timeline** – Visualize recent releases

### Mobile Support
- Works on desktop, tablet, and mobile
- Touch-friendly navigation
- Fast loading with minimal dependencies

### Data Export
- **JSON API** – Access raw metrics for external tools
- **Badges** – Ready-to-use status badges for READMEs
- **Historical Data** – Trend analysis (planned feature)

## 🔧 Technical Implementation

Built with:

- **GitHub Actions** – Scheduled data collection
- **GitHub API** – Fetching live repo and workflow data
- **GitHub Pages** – Hosting the static dashboard
- **HTML/CSS** – Clean, responsive interface

### Data Collection Process
1. **Gather Metrics** – Collect data via GitHub Actions
2. **Process Data** – Calculate health scores
3. **Generate Dashboard** – Build HTML with latest info
4. **Deploy** – Push updated dashboard to GitHub Pages

## 🤝 Contributing

Ways to help:

1. **Suggest Metrics** – Open an issue to propose new ones
2. **Report Bugs** – Let us know about problems or UX issues
3. **Submit Code** – PRs welcome for enhancements
4. **Improve Docs** – Help keep this page clear and useful

## 📞 Support

For questions or feedback:

- **GitHub Issues** – [Report problems or suggestions](https://github.com/twardoch/uubed/issues)
- **GitHub Discussions** – [Ask questions or share thoughts](https://github.com/twardoch/uubed/discussions)

---

*Note: This dashboard gives a top-level view of project health. For detailed insights, check the individual repositories linked above.*