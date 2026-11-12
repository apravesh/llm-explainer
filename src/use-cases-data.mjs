// Scripted illustrations for the "What are LLMs used for?" page. Nothing here is live model output:
// each example is hand-written to show the SHAPE of the text a model receives and continues.
// who: 'app' (text the application adds), 'user' (what the person typed), 'model' (what the LLM writes),
// 'tool' (a result the application pastes in after running something for the model).

export const USE_CASES = Object.freeze([
 {key: 'chat', label: 'Chatbot', kicker: 'ASK AND ANSWER',
  give: 'A question or request.', writes: 'A reply, then stops.', where: 'Assistants such as ChatGPT, Claude and Gemini.',
  point: 'The app wraps your message in a script, exactly as on the previous page. The reply is the continuation.',
  segments: [
   {who: 'app', text: 'User: '}, {who: 'user', text: 'Suggest a name for a small bakery.'}, {who: 'app', text: '\nAssistant: '},
   {who: 'model', text: 'How about “Rise & Shine Bakery”? It sounds warm and hints at fresh bread every morning.'}]},
 {key: 'writing', label: 'Creative writing', kicker: 'CONTINUE MY TEXT',
  give: 'The opening of a story, email or article.', writes: 'What could plausibly come next.', where: 'Writing tools and editors with a “continue” or “draft” button.',
  point: 'No script is needed. Your text is the beginning, and the model simply keeps going in the same voice.',
  segments: [
   {who: 'user', text: 'The lighthouse keeper had not seen a ship in forty days.'},
   {who: 'model', text: ' Then, one grey morning, a small red sail appeared on the horizon, moving steadily against the wind.'}]},
 {key: 'code', label: 'Code completion', kicker: 'FINISH THIS LINE',
  give: 'The code above your cursor.', writes: 'The next line or block of code.', where: 'Editor assistants such as GitHub Copilot.',
  point: 'Code is text too. The editor sends what you have typed so far, and shows the continuation as a grey suggestion you can accept.',
  segments: [
   {who: 'user', text: 'def is_even(n):\n    """Return True if n is even."""\n'},
   {who: 'model', text: '    return n % 2 == 0'}]},
 {key: 'agent', label: 'Agentic workflow', kicker: 'DO A TASK IN STEPS',
  give: 'A goal, and a list of tools it may ask for.', writes: 'One tool request at a time, then a summary.', where: 'Coding agents, research assistants and task automation.',
  point: 'The model cannot run anything itself. It writes a request, the app runs it and pastes the result into the text, and the model continues from there. That loop is what “agent” means.',
  segments: [
   {who: 'app', text: 'Task: '}, {who: 'user', text: 'Fix the failing test.'}, {who: 'app', text: '\nTools: run_tests, read_file, edit_file\nAssistant: '},
   {who: 'model', text: 'run_tests()', kind: 'action'}, {who: 'app', text: '\nTool: '}, {who: 'tool', text: '1 failed: test_total expected 10, got 9'}, {who: 'app', text: '\nAssistant: '},
   {who: 'model', text: 'read_file("cart.py")', kind: 'action'}, {who: 'app', text: '\nTool: '}, {who: 'tool', text: 'total = sum(prices) - 1'}, {who: 'app', text: '\nAssistant: '},
   {who: 'model', text: 'edit_file("cart.py", "total = sum(prices)")', kind: 'action'}, {who: 'app', text: '\nTool: '}, {who: 'tool', text: 'saved'}, {who: 'app', text: '\nAssistant: '},
   {who: 'model', text: 'run_tests()', kind: 'action'}, {who: 'app', text: '\nTool: '}, {who: 'tool', text: 'all 12 tests passed'}, {who: 'app', text: '\nAssistant: '},
   {who: 'model', text: 'Done. The total was one too low; I removed the stray “- 1” in cart.py.', kind: 'final'}]}
].map(Object.freeze));

/** Split text into typing units: each word keeps the whitespace before it, so joining restores the text exactly. */
export const typingUnits = text => text.match(/\s*\S+|\s+$/g) || [];

// ---- Chat with web search: a made-up town and race, so nothing here can be mistaken for real news. ----
export const SEARCH_DEMO = Object.freeze({
 question: 'Who won the Riverton city marathon yesterday?',
 query: 'Riverton city marathon winner',
 results: Object.freeze([
  {site: 'rivertonherald.example', title: 'Okafor wins Riverton Marathon in record time', snippet: 'Amara Okafor won Sunday’s Riverton Marathon in 2:31:07, the fastest time in the race’s history.'},
  {site: 'riverton.example/news', title: 'Marathon weekend: road closures and results', snippet: 'More than 4,000 runners finished the Riverton Marathon on Sunday under clear skies.'},
  {site: 'runlocal.example', title: 'Course record falls at Riverton', snippet: 'Okafor beat the previous course record by 40 seconds, pulling away in the final five kilometres.'}
 ]),
 answer: 'Amara Okafor won the Riverton Marathon, finishing in 2:31:07. That beat the course record by 40 seconds.',
 sources: Object.freeze([1, 3]),
 answerWithoutSearch: 'I do not have information about yesterday’s race. What I know comes from text I was trained on, which ends before then.'
});

/** Step 1 of search: the app asks the model to turn the question into a search query. */
export const queryPrompt = question => [
 {who: 'app', text: 'Write a web search query that would help answer the question.\nQuestion: '}, {who: 'user', text: question}, {who: 'app', text: '\nSearch query: '}];

/** Step 2: the app pastes the search results into the text, above the question. This is the whole trick behind RAG. */
export function searchPrompt(question, results) {
 if (!Array.isArray(results)) throw new TypeError('results must be a list');
 const segments = [];
 if (results.length) {
  segments.push({who: 'app', text: 'Answer using the search results below. Cite them by number.\nSearch results:\n'});
  results.forEach((r, i) => segments.push({who: 'tool', text: `[${i + 1}] ${r.title} (${r.site}): ${r.snippet}\n`}));
 }
 segments.push({who: 'app', text: 'User: '}, {who: 'user', text: question}, {who: 'app', text: '\nAssistant: '});
 return segments;
}
export const segmentsText = segments => segments.map(s => s.text).join('');
