## Description

<!--
Add a comprehensive description of proposed changes

List associated issue number(s) if exist(s)

List associated documentation and benchmarks PR(s) if needed
-->

---

<!--
PR should start as a draft, then move to ready for review state
after CI is passed and all applicable checkboxes are closed.
This approach ensures that reviewers don't spend extra time asking for regular requirements.

You can remove a checkbox as not applicable only if it doesn't relate to this PR in any way.
For example, a PR with docs update doesn't require checkboxes for performance
while a PR with any change in actual code should list checkboxes and
justify how this code change is expected to affect performance (or justification should be self-evident).
-->

<details>
<summary>Checklist:</summary>

**Completeness and readability**

- [ ] I have commented my code, particularly in hard-to-understand areas.
- [ ] I have updated the documentation to reflect the changes or created a separate PR with updates and provided its number in the description, if necessary.
- [ ] Git commit message contains an appropriate signed-off-by string _(see [CONTRIBUTING.md](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/doc/sources/contribute.rst) for details)_.
- [ ] I have resolved any merge conflicts that might occur with the base branch.

**Testing**

- [ ] I have run it locally and tested the changes extensively.
- [ ] All CI jobs are green or I have provided justification why they aren't.
- [ ] I have extended testing suite if new functionality was introduced in this PR.

**Performance**

- [ ] I have measured performance for affected algorithms using [scikit-learn_bench](https://github.com/IntelPython/scikit-learn_bench) and provided at least a summary table with measured data, if performance change is expected.
- [ ] I have provided justification why performance and/or quality metrics have changed or why changes are not expected.
- [ ] I have extended the benchmarking suite and provided a corresponding scikit-learn_bench PR if new measurable functionality was introduced in this PR.

</details>
