# Community Metrics Tracking

The uubed project automatically tracks metrics related to community engagement and repository health. This provides visibility into project growth and helps identify areas needing attention.

## What We Track

### Repository Health
- **Stars & Forks**: Measures interest and adoption
- **Contributors**: Active developer community size
- **Commit Activity**: Development pace and consistency
- **Issue Management**: Response time and resolution rate
- **Pull Request Flow**: Code review throughput

### Engagement Metrics
- **Recent Activity**: Commits, issues, PRs in the last 30 days
- **Community Growth**: New contributors and their contributions
- **Release Cadence**: Frequency and stability of releases
- **Documentation Usage**: Page views and interaction with docs

### Project Quality Indicators
- **Code Health**: Ratio of open to closed issues
- **Maintenance**: Time since last release or commit
- **Community Support**: Average issue response times
- **Development Momentum**: Consistent activity patterns

## Health Score Calculation

Each repository receives a health score (0–100) based on:

- **Recent Activity (30 points)**: Daily commits over the past month
- **Community Engagement (25 points)**: Contributor count and issue resolution
- **Project Popularity (20 points)**: Stars and overall community interest
- **Code Quality (15 points)**: Issue management and PR responsiveness
- **Release Management (10 points)**: Consistency and recency of releases

### Score Interpretation
- **75–100**: Healthy, active project
- **50–74**: Stable but moderately active
- **0–49**: Needs attention—low activity or poor engagement

## Collection Schedule

- **Daily**: Basic metrics collected at 6 AM UTC
- **Weekly**: Aggregated analysis and trend updates
- **Monthly**: Full health reports generated
- **Manual**: Available on demand via workflow trigger

## Metrics Dashboard

Data feeds into the [Project Dashboard](./DASHBOARD.md), offering:

- **Real-time Health Status**: Current scores for each repository
- **Activity Trends**: Visualized development velocity
- **Community Growth**: Historical tracking of stars, forks, and contributors
- **Issue Management**: Open/closed ratios and average response times

## How to Access Metrics

### Latest Snapshot
Current metrics are stored in `community_metrics_latest.json` and used by the dashboard.

### Historical Data
- **GitHub Actions Artifacts**: JSON and CSV files from each collection
- **Retention**: 90 days of data
- **Formats**: Machine-readable (JSON/CSV) and human-readable (Markdown)

### Manual Collection
To run metrics collection manually:
1. Go to the [Actions tab](../../actions/workflows/community-metrics.yml)
2. Click “Run workflow”
3. Download results from the generated artifacts

## Current Repositories

| Repository | Focus | Language | Status |
|------------|-------|----------|--------|
| [uubed](https://github.com/twardoch/uubed) | Project coordination | Markdown/Python | ![Tracking](https://img.shields.io/badge/tracking-active-green) |
| [uubed-rs](https://github.com/twardoch/uubed-rs) | High-performance core | Rust | ![Tracking](https://img.shields.io/badge/tracking-active-green) |
| [uubed-py](https://github.com/twardoch/uubed-py) | Python bindings | Python | ![Tracking](https://img.shields.io/badge/tracking-active-green) |
| [uubed-docs](https://github.com/twardoch/uubed-docs) | Documentation | Markdown | ![Tracking](https://img.shields.io/badge/tracking-active-green) |

## Using Metrics for Decision Making

### For Maintainers
- **Resource Allocation**: Prioritize repositories with declining health
- **Community Engagement**: Spot opportunities to interact more effectively
- **Release Planning**: Assess readiness using activity and issue resolution data
- **Feature Prioritization**: Align with community feedback trends

### For Contributors
- **Contribution Opportunities**: See where help is most needed
- **Project Health**: Understand stability and momentum
- **Impact Tracking**: Measure how your work influences health scores
- **Recognition**: Contributors appear in periodic reports

### For Users
- **Project Stability**: Health scores reflect maintenance quality
- **Support Responsiveness**: Activity levels hint at issue turnaround time
- **Upgrade Planning**: Release cadence helps predict version availability
- **Trustworthiness**: Consistent metrics suggest reliable governance

## Technical Implementation

### Data Collection
```yaml
# Scheduled via GitHub Actions
schedule:
  - cron: '0 6 * * *'  # Runs daily at 6 AM UTC

permissions:
  contents: read
  actions: read
```

### Metrics Storage
- **Formats**: JSON (processing), CSV (analysis), Markdown (reports)
- **Location**: GitHub Actions artifacts with 90-day retention
- **Access**: Publicly available through GitHub API and dashboard

### Privacy Considerations
- **Public Data Only**: All metrics use only public GitHub API data
- **No Personal Info**: Individual contributor details are aggregated
- **Transparent Process**: Scripts and methodology are open source

## Questions or Feedback

- **GitHub Issues**: [Report problems or suggest improvements](../../issues)
- **GitHub Discussions**: [Ask about metrics](../../discussions)
- **Documentation**: [Propose changes to this guide](../../pulls)

## Contributing to Metrics

Improve how we track community health:

1. **Suggest New Metrics**: What’s missing?
2. **Refine Scoring**: Better algorithms for health assessment?
3. **Visualization Ideas**: Enhance dashboard clarity
4. **Trend Analysis**: Help interpret long-term patterns

### Ideas for Improvement
- [ ] Contributor diversity tracking
- [ ] Code quality signals (test coverage, linter stats)
- [ ] Documentation completeness metrics
- [ ] Performance regression monitoring
- [ ] Dependency health across repositories