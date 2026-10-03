import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { analyzeCode, ApiError } from './api'
import { MOCK_ANALYSIS } from './testFixtures'

vi.mock('./api', async () => {
  const actual = await vi.importActual('./api')
  return { ...actual, analyzeCode: vi.fn() }
})

beforeEach(() => {
  analyzeCode.mockReset()
})

async function submitSource(text = 'class Foo:\n    def bar(self): pass\n') {
  const textarea = screen.getByPlaceholderText(/class OrderProcessor/)
  await userEvent.type(textarea, text)
  await userEvent.click(screen.getByRole('button', { name: /analyze code/i }))
}

describe('App end-to-end flow', () => {
  it('goes from input to a populated results page on a successful analysis', async () => {
    analyzeCode.mockResolvedValueOnce(MOCK_ANALYSIS)
    render(<App />)

    await submitSource()

    await waitFor(() => expect(screen.getByText(/analysis results/i)).toBeInTheDocument())
    expect(analyzeCode).toHaveBeenCalledTimes(1)
    expect(screen.getByText('school.py')).toBeInTheDocument()
    expect(screen.getByText('Course.get_marks')).toBeInTheDocument()
  })

  it('shows the loading state while the request is in flight', async () => {
    let resolveFn
    analyzeCode.mockReturnValueOnce(new Promise((resolve) => { resolveFn = resolve }))
    render(<App />)

    await submitSource()
    expect(screen.getByRole('button', { name: /analyzing/i })).toBeDisabled()

    resolveFn(MOCK_ANALYSIS)
    await waitFor(() => expect(screen.getByText(/analysis results/i)).toBeInTheDocument())
  })

  it('shows an error banner and stays on the input page when the API call fails', async () => {
    analyzeCode.mockRejectedValueOnce(new ApiError('parse error: invalid syntax', 400))
    render(<App />)

    await submitSource()

    expect(await screen.findByRole('alert')).toHaveTextContent(/invalid syntax/i)
    expect(screen.queryByText(/analysis results/i)).not.toBeInTheDocument()
  })

  it('returns to the input page from results via "Analyze Another File"', async () => {
    analyzeCode.mockResolvedValueOnce(MOCK_ANALYSIS)
    render(<App />)
    await submitSource()
    await waitFor(() => expect(screen.getByText(/analysis results/i)).toBeInTheDocument())

    await userEvent.click(screen.getByRole('button', { name: /analyze another file/i }))
    expect(screen.getByRole('button', { name: /analyze code/i })).toBeInTheDocument()
  })
})
