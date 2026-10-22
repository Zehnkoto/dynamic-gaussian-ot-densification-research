const papers = window.PAPERS || [];
const grid = document.querySelector('#paperGrid');
const filters = document.querySelector('#filters');
const search = document.querySelector('#paperSearch');
const empty = document.querySelector('#emptyState');
const count = document.querySelector('#paperCount');
let activeTag = '全部';

count.textContent = papers.length;
const tags = ['全部', ...new Set(papers.flatMap(p => p.tags))];

function escapeHTML(value) {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

function renderFilters() {
  filters.innerHTML = tags.map(tag => `<button type="button" class="filter ${tag === activeTag ? 'active' : ''}" data-tag="${escapeHTML(tag)}">${escapeHTML(tag)}</button>`).join('');
}

function renderPapers() {
  const query = search.value.trim().toLowerCase();
  const visible = papers.filter(p => (activeTag === '全部' || p.tags.includes(activeTag)) && [p.title,p.short,p.authors,p.note,...p.tags].join(' ').toLowerCase().includes(query));
  grid.innerHTML = visible.map((p, index) => `
    <article class="paper-card" style="--delay:${index * 35}ms">
      <div class="paper-meta"><span>${escapeHTML(p.year)}</span><span>${escapeHTML(p.short)}</span></div>
      <h3>${escapeHTML(p.title)}</h3>
      <p class="authors">${escapeHTML(p.authors)}</p>
      <p>${escapeHTML(p.note)}</p>
      <div class="paper-tags">${p.tags.map(t => `<span>${escapeHTML(t)}</span>`).join('')}</div>
      <div class="paper-links"><a href="${p.paper}" target="_blank" rel="noopener">论文 ↗</a>${p.code ? `<a href="${p.code}" target="_blank" rel="noopener">代码 ↗</a>` : ''}</div>
    </article>`).join('');
  empty.hidden = visible.length !== 0;
}

filters.addEventListener('click', event => {
  const button = event.target.closest('[data-tag]');
  if (!button) return;
  activeTag = button.dataset.tag;
  renderFilters();
  renderPapers();
});
search.addEventListener('input', renderPapers);

const root = document.documentElement;
const savedTheme = localStorage.getItem('research-theme');
if (savedTheme) root.dataset.theme = savedTheme;
document.querySelector('#themeToggle').addEventListener('click', () => {
  const next = root.dataset.theme === 'light' ? 'dark' : 'light';
  root.dataset.theme = next;
  localStorage.setItem('research-theme', next);
});

const observer = new IntersectionObserver(entries => entries.forEach(entry => {
  if (entry.isIntersecting) entry.target.classList.add('visible');
}), {threshold: .08});
document.querySelectorAll('.phase,.verdict,.formula-card,.quadrant').forEach(el => observer.observe(el));

renderFilters();
renderPapers();
