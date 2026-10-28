# Stack Overflow Developer Survey 2023–2026: AI questions (harmonised)

This dataset comes from the [Stack Overflow Developer Survey](https://survey.stackoverflow.co/), an online survey of developers that Stack Overflow runs every year and publishes as open data under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/). I combined the three editions that ask about AI, 2023, 2024 and 2025, into one file. The 2026 edition is described separately below, because many of its questions changed.

The file has 203,812 responses: 89,184 from 2023, 65,437 from 2024 and 49,191 from 2025. The group I plan to study is professional developers living in the EU: 23,788 in 2023, 15,690 in 2024 and 9,758 in 2025.

The table lists each variable, the question it comes from, and the share of those EU developers who answered it in each year.

| Variable | Survey question | Answers | Answered (%) 2023 / 2024 / 2025 |
|---|---|---|---|
| Year | Survey edition | 2023, 2024, 2025 | 100 / 100 / 100 |
| MainBranch, ProDev | Are you someone who writes code? Which option best describes you today? | Survey options; professional developer yes or no | 100 / 100 / 100 |
| Country, EU | Where do you live? | Country; EU-27 yes or no | 100 / 100 / 100 |
| Age | What is your age? | 7 age groups | 100 / 100 / 100 |
| WorkExp, Cohort | How many years of professional work experience do you have? | Years; groups of 0–4, 5–9, 10–19, 20+ | 57 / 60 / 98 |
| EdLevel | What is the highest level of formal education you've completed? | 5 groups | 98 / 99 / 99 |
| Employment | What best describes your current employment status? | Employed, self-employed, student, not employed, retired | 100 / 100 / 100 |
| Remote | What best describes your current work situation? | Remote, hybrid, in-person | 95 / 95 / 85 |
| OrgSize | Approximately how many people work for your employer? | 5 size groups | 87 / 93 / 84 |
| ICorPM | Are you an individual contributor or people manager? | 2 options | 57 / 61 / 83 |
| AIUse, AIUseFreq | Do you currently use AI tools in your development process? | Use, plan to, no; daily, weekly or monthly in 2025 | 100 / 99 / 97 |
| AITaskN, AITaskN8 | Which parts of your development workflow do you use AI tools for? | Number of tasks; the same count on the 8 tasks asked in every year | 42 / 59 / 90 |
| AISent, AISentScore | How favourable is your stance on using AI tools in your workflow? | 5 options; score 1 to 5 | 67 / 73 / 96 |
| AITrust, AITrustScore | How much do you trust the accuracy of AI tools' output? | 5 options; score 1 to 5 | 67 / 60 / 96 |
| AIBenefits | What are the most important benefits you hope to get from AI tools? | Several can be chosen (2023 and 2024 only) | 42 / 59 / – |
| AIThreat | Do you believe AI is a threat to your current job? | Yes, no, not sure (2024 and 2025 only) | – / 72 / 100 |
| AIOpen | Looking ahead 3–5 years, what skills will remain valuable for developers as AI tools become more capable? | Open text (2025 only) | – / – / 66 |

## 2026 edition

The 2026 survey was published in October 2026 and has 30,903 responses. It no longer asks whether you are a developer by profession, so I define professional developers as people who create code as a primary part of their job and are employed or self-employed. That gives 5,363 professional developers living in the EU.

Several questions changed. The question on whether AI is a threat to your job was dropped. The list of AI tasks was rewritten, with 19 tasks and code generation split into familiar and unfamiliar areas. New questions ask how AI changed satisfaction with each kind of task, whether and why people avoid AI, whether they learned AI tooling in the past year, and how their organisation approaches AI.

Before any 2026 answers were analysed, 30% of the responses were set aside at random and sealed, so the final hypotheses can be tested on data nobody has looked at. The answer rates below come from the other 70% (3,743 EU professional developers).

| Variable | Survey question | Answers | Answered (%) 2026 |
|---|---|---|---|
| CodeMain, Employment | Are you someone who creates code? What is your employment status? | Used to define professional developers | 100 |
| Country | Where do you live? | Country | 100 |
| Age | What is your age? | Age groups | 100 |
| WorkExp | How many years have you been working professionally in total? | Years | 99 |
| YearsCode | Including any education, how many years have you been coding in total? | Years | 99 |
| Education | What is the highest level of formal education you've completed? | Education groups | 100 |
| WorkSetting | Which best describes your current work situation? | Remote, hybrid, in-person | 100 |
| OrgSize | Approximately how many people are employed by your employer? | Size groups | 100 |
| ICorPM | Are you an individual contributor or people manager? | 2 options | 99 |
| LearnAI | In the last year, did you spend time learning AI programming or AI-enabled tooling? | Yes on my own, yes at work, both, no | 100 |
| AISelect | Do you currently use AI tools or AI agents in your role at work? | Types of AI tool used, or none | 82 |
| AIFreq | How often do you use each of these AI tools or agents at work? | Daily, weekly, monthly or less, per tool type | 60 |
| AIDaily | How much of your workday do you use AI tools? (daily users) | Under 1 hour to 4+ hours | 56 |
| AISentiment | How would you describe your attitude toward using AI tools or agents at work? | 5 options | 72 |
| AITrust | How much do you trust the output from AI tools or agents? | 5 options | 72 |
| AIComplex | How well do AI tools or agents handle complex tasks? | 5 options | 72 |
| AIWorkCurrentlyUse | Which parts of your workflow do you use AI for, and will that change? | 19 tasks; use now, expect more, expect less, do not plan to use | 56 |
| KnowAgentDelegate | Which tasks have you delegated to an AI tool or agent in the last 30 days? | 17 tasks | 78 |
| AIWorkNext | What do you usually do before using an AI-generated answer? | For example use as-is, run it locally, read the docs | 66 |
| AIImpact | How useful has AI been for each task? | 7 task types; very useful to not useful | about 64 |
| AISat | Has AI increased or decreased your satisfaction with each task? | 7 task types; greatly improved to greatly decreased | about 61 |
| AIAvoid | Do you try to avoid using AI tools or agents at work? | No, or the main reason (for example losing job skills, ethics, privacy) | 73 |
| AIUse | How are AI tools or agents used at your organisation? | From optional to measured in performance reviews | 74 |
| AITry | How does your organisation approach AI tools or models? | For example one approved tool, teams choose their own | 74 |

## Files

| File | Contents |
|---|---|
| `so_ai_harmonised_2023_2025.zip` | Harmonised 2023–2025 file (203,812 responses, variables in the first table) |
| `so2026_explore_70.zip` | 2026 exploration sample: a random 70% of the 2026 responses (21,563), all original variables |
| `holdout_2026_SPLIT_PROTOCOL.txt` | How the 2026 file was split into the 70% sample and the sealed 30% holdout |
| `holdout_2026_SHA256SUMS.txt` | Fingerprints of the split files, so anyone can check the holdout was not changed |

The sealed 30% holdout is not included here. It will be added once the hypotheses have been tested on it. The full original survey files are available from [Stack Overflow](https://survey.stackoverflow.co/).

## Licence

The survey data are © Stack Exchange Inc. and made available under the Open Database License (ODbL) v1.0. Any derived dataset shared publicly must also be shared under the ODbL, with attribution to Stack Overflow.
