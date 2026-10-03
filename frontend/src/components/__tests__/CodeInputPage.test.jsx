import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import CodeInputPage from '../CodeInputPage'

describe('CodeInputPage', () => {
  it('rejects analyzing empty input without calling onAnalyze', async () => {
    const onAnalyze = vi.fn()
    render(<CodeInputPage onAnalyze={onAnalyze} loading={false} apiError={null} />)

    await userEvent.click(screen.getByRole('button', { name: /analyze code/i }))

    expect(onAnalyze).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent(/paste some python code/i)
  })

  it('submits pasted source and filename', async () => {
    const onAnalyze = vi.fn()
    render(<CodeInputPage onAnalyze={onAnalyze} loading={false} apiError={null} />)

    const textarea = screen.getByPlaceholderText(/class OrderProcessor/)
    await userEvent.type(textarea, 'def f(): pass')

    const filenameInput = screen.getByLabelText(/filename/i)
    await userEvent.clear(filenameInput)
    await userEvent.type(filenameInput, 'my_module.py')

    await userEvent.click(screen.getByRole('button', { name: /analyze code/i }))

    expect(onAnalyze).toHaveBeenCalledWith('def f(): pass', 'my_module.py')
  })

  it('rejects a non-.py file upload with an inline error', async () => {
    const onAnalyze = vi.fn()
    render(<CodeInputPage onAnalyze={onAnalyze} loading={false} apiError={null} />)

    const file = new File(['not python'], 'notes.txt', { type: 'text/plain' })
    const fileInput = document.querySelector('input[type="file"]')
    fireEvent.change(fileInput, { target: { files: [file] } })

    expect(await screen.findByRole('alert')).toHaveTextContent(/not a \.py file/i)
    expect(onAnalyze).not.toHaveBeenCalled()
  })

  it('loads a valid .py file into the textarea and filename field', async () => {
    render(<CodeInputPage onAnalyze={vi.fn()} loading={false} apiError={null} />)

    const file = new File(['class Foo:\n    pass\n'], 'foo.py', { type: 'text/x-python' })
    const fileInput = document.querySelector('input[type="file"]')
    fireEvent.change(fileInput, { target: { files: [file] } })

    expect(await screen.findByDisplayValue(/class Foo/)).toBeInTheDocument()
    expect(screen.getByLabelText(/filename/i)).toHaveValue('foo.py')
  })

  it('shows a spinner label and disables the button while loading', () => {
    render(<CodeInputPage onAnalyze={vi.fn()} loading={true} apiError={null} />)
    const button = screen.getByRole('button', { name: /analyzing/i })
    expect(button).toBeDisabled()
  })

  it('surfaces an API error passed from the parent', () => {
    render(<CodeInputPage onAnalyze={vi.fn()} loading={false} apiError="network down" />)
    expect(screen.getByRole('alert')).toHaveTextContent('network down')
  })
})
