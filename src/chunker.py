"""
Slide-wise chunking strategy that preserves project-level context.
Each project becomes one primary chunk with structured metadata for filtering.
Additionally creates sub-chunks for specific aspects (problem, solution, tech, outcomes)
to improve retrieval precision.
"""
import re


def create_project_chunk(project: dict) -> dict:
    """
    Create a rich, structured chunk from a project record.
    This is the PRIMARY chunk - one per project, containing everything.
    """
    techs = ", ".join(project['technologies']) if project['technologies'] else "Not specified"

    # Build a clean, structured text representation optimized for embedding
    chunk_text = f"""Project: {project['project_id']} - {project['title']}
Domain: {project['domain']}
Type: {project['project_type']}
Team Size: {project['team_size']} people
Duration: {project['duration_months']} months
Technologies: {techs}

{project['full_text']}"""

    return {
        'id': f"{project['project_id']}_full",
        'text': chunk_text,
        'metadata': {
            'project_id': project['project_id'],
            'title': project['title'],
            'domain': project['domain'],
            'team_size': project['team_size'],
            'duration_months': project['duration_months'],
            'project_type': project['project_type'],
            'technologies': techs,
            'slide_numbers': ",".join(map(str, project['slide_numbers'])),
            'chunk_type': 'full_project',
        }
    }


def extract_section(text: str, section_name: str) -> str:
    """Extract a named section from project text."""
    patterns = {
        'problem': r'PROBLEM\s*(?:STATEMENT|CONTEXT)\s*\n(.*?)(?=SOLUTION|PROJECT|TECHNOLOGY|FULL|KEY|$)',
        'solution': r'SOLUTION\s*APPROACH\s*\n(.*?)(?=PROJECT|TECHNOLOGY|FULL|KEY|EXPECTED|$)',
        'outcomes': r'(?:EXPECTED\s*OUTCOMES|KEY\s*METRICS.*?IMPACT)\s*\n(.*?)(?=PROBLEM|SOLUTION|PROJECT|TECHNOLOGY|FULL|$)',
    }
    pattern = patterns.get(section_name, '')
    if not pattern:
        return ''
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ''


def create_aspect_chunks(project: dict) -> list[dict]:
    """
    Create sub-chunks for specific project aspects.
    These improve retrieval for targeted queries like
    "what technologies were used" or "what problems were solved".
    """
    chunks = []
    text = project['full_text']
    base_meta = {
        'project_id': project['project_id'],
        'title': project['title'],
        'domain': project['domain'],
        'team_size': project['team_size'],
        'duration_months': project['duration_months'],
        'project_type': project['project_type'],
        'technologies': ", ".join(project['technologies']) if project['technologies'] else "Not specified",
        'slide_numbers': ",".join(map(str, project['slide_numbers'])),
    }

    # Problem chunk
    problem = extract_section(text, 'problem')
    if problem:
        chunks.append({
            'id': f"{project['project_id']}_problem",
            'text': f"Project: {project['project_id']} - {project['title']}\nDomain: {project['domain']}\n\nProblem Statement:\n{problem}",
            'metadata': {**base_meta, 'chunk_type': 'problem'},
        })

    # Solution chunk
    solution = extract_section(text, 'solution')
    if solution:
        chunks.append({
            'id': f"{project['project_id']}_solution",
            'text': f"Project: {project['project_id']} - {project['title']}\nDomain: {project['domain']}\nTechnologies: {base_meta['technologies']}\n\nSolution Approach:\n{solution}",
            'metadata': {**base_meta, 'chunk_type': 'solution'},
        })

    # Outcomes chunk
    outcomes = extract_section(text, 'outcomes')
    if outcomes:
        chunks.append({
            'id': f"{project['project_id']}_outcomes",
            'text': f"Project: {project['project_id']} - {project['title']}\nDomain: {project['domain']}\n\nExpected Outcomes & Impact:\n{outcomes}",
            'metadata': {**base_meta, 'chunk_type': 'outcomes'},
        })

    return chunks


def chunk_projects(projects: list[dict]) -> list[dict]:
    """
    Main chunking function. Creates:
    1. One full project chunk per project (for comprehensive retrieval)
    2. Aspect-level sub-chunks (problem, solution, outcomes) for precision
    """
    all_chunks = []
    for project in projects:
        # Primary full-project chunk
        all_chunks.append(create_project_chunk(project))
        # Aspect sub-chunks
        all_chunks.extend(create_aspect_chunks(project))

    return all_chunks


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    from slide_parser import parse_pptx
    projects = parse_pptx("D:/hackathone2/Dataset_project_repository.pptx")
    chunks = chunk_projects(projects)
    print(f"Total chunks: {len(chunks)}")

    # Count by type
    from collections import Counter
    types = Counter(c['metadata']['chunk_type'] for c in chunks)
    for t, count in types.items():
        print(f"  {t}: {count}")

    print(f"\nSample chunk (P001 full):")
    for c in chunks:
        if c['id'] == 'P001_full':
            print(c['text'][:500])
            break
