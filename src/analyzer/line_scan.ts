/**
 * newline iteration without String.split (large-file friendly).
 */

export function forEachLineInRange(
  s: string,
  start: number,
  end: number,
  fn: (line: string) => void
): void {
  let i = start;
  const e = Math.min(end, s.length);
  while (i < e) {
    const nl = s.indexOf("\n", i);
    const lineEnd = nl === -1 ? e : Math.min(nl, e);
    let line = s.slice(i, lineEnd);
    if (line.charCodeAt(line.length - 1) === 13) {
      line = line.slice(0, -1);
    }
    fn(line);
    if (nl === -1 || nl >= e) {
      break;
    }
    i = nl + 1;
  }
}
