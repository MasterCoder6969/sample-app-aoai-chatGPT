import { useRef, useState, useEffect, useContext, useLayoutEffect } from 'react'
import { CommandBarButton, IconButton, Dialog, DialogType, Stack } from '@fluentui/react'
import { SquareRegular, ShieldLockRegular, ErrorCircleRegular } from '@fluentui/react-icons'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import uuid from 'react-uuid'
import { isEmpty } from 'lodash'
import DOMPurify from 'dompurify'

import styles from './Chat.module.css'
import Contoso from '../../assets/MadridLogo.png'
import { XSSAllowTags } from '../../constants/xssAllowTags'

import {
  UpdateFeedback,
  StoreCode,
  ChatMessage,
  ConversationRequest,
  conversationApi,
  Citation,
  ToolMessageContent,
  AzureSqlServerExecResults,
  ChatResponse,
  getUserInfo,
  Conversation,
  historyGenerate,
  historyUpdate,
  historyClear,
  ChatHistoryLoadingState,
  CosmosDBStatus,
  ErrorMessage,
  ExecResults,
  AzureSqlServerCodeExecResult
} from "../../api";
import { Answer } from "../../components/Answer";
import { QuestionInput } from "../../components/QuestionInput";
import { ChatHistoryPanel } from "../../components/ChatHistory/ChatHistoryPanel";
import { AppStateContext } from "../../state/AppProvider";
import { useBoolean } from "@fluentui/react-hooks";

const enum messageStatus {
  NotRunning = 'Not Running',
  Processing = 'Processing',
  Done = 'Done'
}
interface DropDownProps {
  categories : string[]
  setSelectedCategory : (category: string) => void
}
const DropDown = ({categories, setSelectedCategory} : DropDownProps) => {
  return (
    <select name="feedback" onChange={e => setSelectedCategory(e.target.value)} className={styles.chatInput}>
      {categories.map((cat:string)=>(<option value={cat}>{cat}</option>))}
    </select>
  );
}

const Chat = () => {
  const appStateContext = useContext(AppStateContext)
  const ui = appStateContext?.state.frontendSettings?.ui
  const AUTH_ENABLED = appStateContext?.state.frontendSettings?.auth_enabled
  const chatMessageStreamEnd = useRef<HTMLDivElement | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [showLoadingMessage, setShowLoadingMessage] = useState<boolean>(false)
  const [activeCitation, setActiveCitation] = useState<Citation>()
  const [isCitationPanelOpen, setIsCitationPanelOpen] = useState<boolean>(false)
  const [isIntentsPanelOpen, setIsIntentsPanelOpen] = useState<boolean>(false)
  const abortFuncs = useRef([] as AbortController[])
  const [showAuthMessage, setShowAuthMessage] = useState<boolean | undefined>()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [oldMessages, setOldMessages] = useState<ChatMessage[]>([])
  const [execResults, setExecResults] = useState<ExecResults[]>([])
  const [processMessages, setProcessMessages] = useState<messageStatus>(messageStatus.NotRunning)
  const [clearingChat, setClearingChat] = useState<boolean>(false)
  const [hideErrorDialog, { toggle: toggleErrorDialog }] = useBoolean(true)
  const [errorMsg, setErrorMsg] = useState<ErrorMessage | null>()
  const [isFeedbackOpen,setIsFeedbackOpen] = useState<boolean>(false)
  const [currentFeedback, setCurrentFeedback] = useState<string>("")
  const [isDropDownCompleted, setIsDropDownCompleted] = useState<boolean>(false)
  const [isFeedbackSent,setIsFeedbackSent] = useState<boolean>(false)
  const [code, setCode] = useState<string>("")
  const [hasCode, setHasCode] = useState<boolean>(false)


  const errorDialogContentProps = {
    type: DialogType.close,
    title: errorMsg?.title,
    closeButtonAriaLabel: 'Close',
    subText: errorMsg?.subtitle
  }

  const modalProps = {
    titleAriaId: 'labelId',
    subtitleAriaId: 'subTextId',
    isBlocking: true,
    styles: { main: { maxWidth: 450 } }
  }

  const [ASSISTANT, TOOL, ERROR] = ['assistant', 'tool', 'error']
  const NO_CONTENT_ERROR = 'No content in messages object.'
  useEffect(() => {
    if (
      appStateContext?.state.isCosmosDBAvailable?.status !== CosmosDBStatus.Working &&
      appStateContext?.state.isCosmosDBAvailable?.status !== CosmosDBStatus.NotConfigured &&
      appStateContext?.state.chatHistoryLoadingState === ChatHistoryLoadingState.Fail &&
      hideErrorDialog
    ) {
      let subtitle = `${appStateContext.state.isCosmosDBAvailable.status}. Please contact the site administrator.`
      setErrorMsg({
        title: 'Chat history is not enabled',
        subtitle: subtitle
      })
      toggleErrorDialog()
    }
  }, [appStateContext?.state.isCosmosDBAvailable])

  const handleErrorDialogClose = () => {
    toggleErrorDialog()
    setTimeout(() => {
      setErrorMsg(null)
    }, 500)
  }

  useEffect(() => {
    setIsLoading(appStateContext?.state.chatHistoryLoadingState === ChatHistoryLoadingState.Loading)
  }, [appStateContext?.state.chatHistoryLoadingState])

  const getUserInfoList = async () => {
    if (!AUTH_ENABLED) {
      setShowAuthMessage(false)
      return
    }
    const userInfoList = await getUserInfo()
    if (userInfoList.length === 0 && window.location.hostname !== '127.0.0.1') {
      setShowAuthMessage(true)
    } else {
      setShowAuthMessage(false)
    }
  }

  let assistantMessage = {} as ChatMessage
  let toolMessage = {} as ChatMessage
  let assistantContent = ''

  const processResultMessage = (resultMessage: ChatMessage, userMessage: ChatMessage, conversationId?: string) => {
    if (resultMessage.content.includes('all_exec_results')) {
      const parsedExecResults = JSON.parse(resultMessage.content) as AzureSqlServerExecResults
      setExecResults(parsedExecResults.all_exec_results)
    }

    if (resultMessage.role === ASSISTANT) {
      assistantContent += resultMessage.content
      assistantMessage = resultMessage
      assistantMessage.content = assistantContent

      if (resultMessage.context) {
        toolMessage = {
          id: uuid(),
          role: TOOL,
          content: resultMessage.context,
          date: new Date().toISOString()
        }
      }
    }

    if (resultMessage.role === TOOL) toolMessage = resultMessage

    if (!conversationId) {
      isEmpty(toolMessage)
        ? setMessages([...messages, userMessage, assistantMessage])
        : setMessages([...messages, userMessage, toolMessage, assistantMessage])
    } else {
      isEmpty(toolMessage)
        ? setMessages([...messages, assistantMessage])
        : setMessages([...messages,toolMessage, assistantMessage])
    }
  }

  const makeApiRequestWithoutCosmosDB = async (question: string, conversationId?: string) => {
    setIsLoading(true)
    setShowLoadingMessage(true)
    const abortController = new AbortController()
    abortFuncs.current.unshift(abortController)

    const userMessage: ChatMessage = {
      id: uuid(),
      role: 'user',
      content: question,
      date: new Date().toISOString()
    }

    let conversation: Conversation | null | undefined
    if (!conversationId) {
      conversation = {
        id: conversationId ?? uuid(),
        title: question,
        messages: [userMessage],
        date: new Date().toISOString()
      }
    } else {
      conversation = appStateContext?.state?.currentChat
      if (!conversation) {
        console.error('Conversation not found.')
        setIsLoading(false)
        setShowLoadingMessage(false)
        abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
        return
      } else {
        conversation.messages.push(userMessage)
      }
    }

    appStateContext?.dispatch({ type: 'UPDATE_CURRENT_CHAT', payload: conversation })
    setMessages(conversation.messages)
    updateFeedback("",code)

    const request: ConversationRequest = {
      messages: [...conversation.messages.filter(answer => answer.role !== ERROR)]
    }

    let result = {} as ChatResponse
    try {
      const response = await conversationApi(request, abortController.signal)
      if (response?.body) {
        const reader = response.body.getReader()

        let runningText = ''
        while (true) {
          setProcessMessages(messageStatus.Processing)
          const { done, value } = await reader.read()
          if (done) break

          var text = new TextDecoder('utf-8').decode(value)
          const objects = text.split('\n')
          objects.forEach(obj => {
            try {
              if (obj !== '' && obj !== '{}') {
                runningText += obj
                result = JSON.parse(runningText)
                if (result.choices?.length > 0) {
                  result.choices[0].messages.forEach(msg => {
                    msg.id = result.id
                    msg.date = new Date().toISOString()
                  })
                  if (result.choices[0].messages?.some(m => m.role === ASSISTANT)) {
                    setShowLoadingMessage(false)
                  }
                  result.choices[0].messages.forEach(resultObj => {
                    processResultMessage(resultObj, userMessage, conversationId)
                  })
                } else if (result.error) {
                  throw Error(result.error)
                }
                runningText = ''
              }
            } catch (e) {
              if (!(e instanceof SyntaxError)) {
                console.error(e)
                throw e
              } else {
                console.log('Incomplete message. Continuing...')
              }
            }
          })
        }
        conversation.messages.push(toolMessage, assistantMessage)
        appStateContext?.dispatch({ type: 'UPDATE_CURRENT_CHAT', payload: conversation })
        setMessages([...messages, toolMessage, assistantMessage])
      }
    } catch (e) {
      if (!abortController.signal.aborted) {
        let errorMessage =
          'Ha habido un error. Porfavor intente de nuevo. Si el problema persiste, porfavor contacte con el administrador.'
        if (result.error?.message) {
          errorMessage = result.error.message
        } else if (typeof result.error === 'string') {
          errorMessage = result.error
        }

        errorMessage = parseErrorMessage(errorMessage)

        let errorChatMsg: ChatMessage = {
          id: uuid(),
          role: ERROR,
          content: errorMessage,
          date: new Date().toISOString()
        }
        conversation.messages.push(errorChatMsg)
        appStateContext?.dispatch({ type: 'UPDATE_CURRENT_CHAT', payload: conversation })
        setMessages([...messages, errorChatMsg])
      } else {
        setMessages([...messages, userMessage])
      }
    } finally {
      setIsLoading(false)
      setShowLoadingMessage(false)
      abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
      setProcessMessages(messageStatus.Done)
      updateFeedback("", code)
    }

    return abortController.abort()
  }

  const makeApiRequestWithCosmosDB = async (question: string, conversationId?: string) => {
    setIsLoading(true)
    setShowLoadingMessage(true)
    const abortController = new AbortController()
    abortFuncs.current.unshift(abortController)

    const userMessage: ChatMessage = {
      id: uuid(),
      role: 'user',
      content: question,
      date: new Date().toISOString()
    }

    //api call params set here (generate)
    let request: ConversationRequest
    let conversation
    if (conversationId) {
      conversation = appStateContext?.state?.chatHistory?.find(conv => conv.id === conversationId)
      if (!conversation) {
        console.error('Conversation not found.')
        setIsLoading(false)
        setShowLoadingMessage(false)
        abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
        return
      } else {
        conversation.messages.push(userMessage)
        request = {
          messages: [...conversation.messages.filter(answer => answer.role !== ERROR)]
        }
      }
    } else {
      request = {
        messages: [userMessage].filter(answer => answer.role !== ERROR)
      }
      setMessages(request.messages)
    }
    let result = {} as ChatResponse
    var errorResponseMessage = 'Ha habido un error. Porfavor intente de nuevo. Si el problema persiste, porfavor contacte con el administrador.'
    try {
      const response = conversationId
        ? await historyGenerate(request, abortController.signal, conversationId)
        : await historyGenerate(request, abortController.signal)
      if (!response?.ok) {
        const responseJson = await response.json()
        errorResponseMessage =
          responseJson.error === undefined ? errorResponseMessage : parseErrorMessage(responseJson.error)
        let errorChatMsg: ChatMessage = {
          id: uuid(),
          role: ERROR,
          content: `Ha habido un error generando la respuesta. El historial de chat no se puede guardar en este momento. ${errorResponseMessage}`,
          date: new Date().toISOString()
        }
        let resultConversation
        if (conversationId) {
          resultConversation = appStateContext?.state?.chatHistory?.find(conv => conv.id === conversationId)
          if (!resultConversation) {
            console.error('Conversation not found.')
            setIsLoading(false)
            setShowLoadingMessage(false)
            abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
            return
          }
          resultConversation.messages.push(errorChatMsg)
        } else {
          setMessages([...messages, userMessage, errorChatMsg])
          setIsLoading(false)
          setShowLoadingMessage(false)
          abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
          return
        }
        appStateContext?.dispatch({ type: 'UPDATE_CURRENT_CHAT', payload: resultConversation })
        setMessages([...resultConversation.messages])
        return
      }
      if (response?.body) {
        const reader = response.body.getReader()

        let runningText = ''
        while (true) {
          setProcessMessages(messageStatus.Processing)
          const { done, value } = await reader.read()
          if (done) break

          var text = new TextDecoder('utf-8').decode(value)
          const objects = text.split('\n')
          objects.forEach(obj => {
            try {
              if (obj !== '' && obj !== '{}') {
                runningText += obj
                result = JSON.parse(runningText)
                if (!result.choices?.[0]?.messages?.[0].content) {
                  errorResponseMessage = NO_CONTENT_ERROR
                  throw Error()
                }
                if (result.choices?.length > 0) {
                  result.choices[0].messages.forEach(msg => {
                    msg.id = result.id
                    msg.date = new Date().toISOString()
                  })
                  if (result.choices[0].messages?.some(m => m.role === ASSISTANT)) {
                    setShowLoadingMessage(false)
                  }
                  result.choices[0].messages.forEach(resultObj => {
                    processResultMessage(resultObj, userMessage, conversationId)
                  })
                }
                runningText = ''
              } else if (result.error) {
                throw Error(result.error)
              }
            } catch (e) {
              if (!(e instanceof SyntaxError)) {
                console.error(e)
                throw e
              } else {
                console.log('Mensaje incompleto. Continuando...')
              }
            }
          })
        }

        let resultConversation
        if (conversationId) {
          resultConversation = appStateContext?.state?.chatHistory?.find(conv => conv.id === conversationId)
          if (!resultConversation) {
            console.error('Conversation not found.')
            setIsLoading(false)
            setShowLoadingMessage(false)
            abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
            return
          }
          isEmpty(toolMessage)
            ? resultConversation.messages.push(assistantMessage)
            : resultConversation.messages.push(toolMessage, assistantMessage)
        } else {
          resultConversation = {
            id: result.history_metadata.conversation_id,
            title: result.history_metadata.title,
            messages: [userMessage],
            date: result.history_metadata.date
          }
          isEmpty(toolMessage)
            ? resultConversation.messages.push(assistantMessage)
            : resultConversation.messages.push(toolMessage, assistantMessage)
        }
        if (!resultConversation) {
          setIsLoading(false)
          setShowLoadingMessage(false)
          abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
          return
        }
        appStateContext?.dispatch({ type: 'UPDATE_CURRENT_CHAT', payload: resultConversation })
        isEmpty(toolMessage)
          ? setMessages([...messages, assistantMessage])
          : setMessages([...messages, toolMessage, assistantMessage])
      }
    } catch (e) {
      if (!abortController.signal.aborted) {
        let errorMessage = `Ha ocurrido un error. ${errorResponseMessage}`
        if (result.error?.message) {
          errorMessage = result.error.message
        } else if (typeof result.error === 'string') {
          errorMessage = result.error
        }

        errorMessage = parseErrorMessage(errorMessage)

        let errorChatMsg: ChatMessage = {
          id: uuid(),
          role: ERROR,
          content: errorMessage,
          date: new Date().toISOString()
        }
        let resultConversation
        if (conversationId) {
          resultConversation = appStateContext?.state?.chatHistory?.find(conv => conv.id === conversationId)
          if (!resultConversation) {
            console.error('Conversation not found.')
            setIsLoading(false)
            setShowLoadingMessage(false)
            abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
            return
          }
          resultConversation.messages.push(errorChatMsg)
        } else {
          if (!result.history_metadata) {
            console.error('Error obteniendo los datos.', result)
            let errorChatMsg: ChatMessage = {
              id: uuid(),
              role: ERROR,
              content: errorMessage,
              date: new Date().toISOString()
            }
            setMessages([...messages, userMessage, errorChatMsg])
            setIsLoading(false)
            setShowLoadingMessage(false)
            abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
            return
          }
          resultConversation = {
            id: result.history_metadata.conversation_id,
            title: result.history_metadata.title,
            messages: [userMessage],
            date: result.history_metadata.date
          }
          resultConversation.messages.push(errorChatMsg)
        }
        if (!resultConversation) {
          setIsLoading(false)
          setShowLoadingMessage(false)
          abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
          return
        }
        appStateContext?.dispatch({ type: 'UPDATE_CURRENT_CHAT', payload: resultConversation })
        setMessages([...messages, errorChatMsg])
      } else {
        setMessages([...messages, userMessage])
      }
    } finally {
      setIsLoading(false)
      setShowLoadingMessage(false)
      abortFuncs.current = abortFuncs.current.filter(a => a !== abortController)
      setProcessMessages(messageStatus.Done)
    }
    return abortController.abort()
  }
  const updateFeedback = async (feedback : string, id?:string) => {
    const abortController = new AbortController()
    abortFuncs.current.unshift(abortController)
    const response = await UpdateFeedback(feedback, messages.map((answer,_)=>(answer.content)), code, abortController.signal)
    if (response) {
      setCurrentFeedback(feedback)
      return
    }
    return abortController.abort()
  }
  const clearChat = async () => {
    setClearingChat(true)
    if (appStateContext?.state.currentChat?.id && appStateContext?.state.isCosmosDBAvailable.cosmosDB) {
      let response = await historyClear(appStateContext?.state.currentChat.id)
      if (!response.ok) {
        setErrorMsg({
          title: 'Error borrando el chat',
          subtitle: 'Porfavor intente de nuevo. Si el problema persiste, porfavor contacte con el administrador.'
        })
        toggleErrorDialog()
      } else {
        appStateContext?.dispatch({
          type: 'DELETE_CURRENT_CHAT_MESSAGES',
          payload: appStateContext?.state.currentChat.id
        })
        appStateContext?.dispatch({ type: 'UPDATE_CHAT_HISTORY', payload: appStateContext?.state.currentChat })
        setActiveCitation(undefined)
        setIsCitationPanelOpen(false)
        setIsIntentsPanelOpen(false)
        setOldMessages(oldMessages.concat(messages))
        setMessages([])
        setHasCode(false)
        setCode("")
        setIsFeedbackOpen(false)
        setCurrentFeedback("")
        setIsFeedbackSent(false)
        setIsDropDownCompleted(false)
      }
    }
    setClearingChat(false)
  }

  const tryGetRaiPrettyError = (errorMessage: string) => {
    try {
      // Using a regex to extract the JSON part that contains "innererror"
      const match = errorMessage.match(/'innererror': ({.*})\}\}/)
      if (match) {
        // Replacing single quotes with double quotes and converting Python-like booleans to JSON booleans
        const fixedJson = match[1]
          .replace(/'/g, '"')
          .replace(/\bTrue\b/g, 'true')
          .replace(/\bFalse\b/g, 'false')
        const innerErrorJson = JSON.parse(fixedJson)
        let reason = ''
        // Check if jailbreak content filter is the reason of the error
        const jailbreak = innerErrorJson.content_filter_result.jailbreak
        if (jailbreak.filtered === true) {
          reason = 'Jailbreak'
        }

        // Returning the prettified error message
        if (reason !== '') {
          return (
            'La prompt fue filtrada por Azure OpenAI’s filtro de contenido.\n' +
            'Razon: Esta prompt contiene contenido considerado ' +
            reason +
            '\n\n' +
            'Porfavor modifique su prompt y vuélvalo a intentar. Más información: https://go.microsoft.com/fwlink/?linkid=2198766'
          )
        }
      }
    } catch (e) {
      console.error('Failed to parse the error:', e)
    }
    return errorMessage
  }

  const parseErrorMessage = (errorMessage: string) => {
    let errorCodeMessage = errorMessage.substring(0, errorMessage.indexOf('-') + 1)
    const innerErrorCue = "{\\'error\\': {\\'message\\': "
    if (errorMessage.includes(innerErrorCue)) {
      try {
        let innerErrorString = errorMessage.substring(errorMessage.indexOf(innerErrorCue))
        if (innerErrorString.endsWith("'}}")) {
          innerErrorString = innerErrorString.substring(0, innerErrorString.length - 3)
        }
        innerErrorString = innerErrorString.replaceAll("\\'", "'")
        let newErrorMessage = errorCodeMessage + ' ' + innerErrorString
        errorMessage = newErrorMessage
      } catch (e) {
        console.error('Error parsing inner error message: ', e)
      }
    }

    return tryGetRaiPrettyError(errorMessage)
  }

  const newChat = () => {
    setOldMessages(oldMessages.concat(messages))
    setProcessMessages(messageStatus.Processing)
    setMessages([])
    setIsCitationPanelOpen(false)
    setIsIntentsPanelOpen(false)
    setActiveCitation(undefined)
    appStateContext?.dispatch({ type: 'UPDATE_CURRENT_CHAT', payload: null })
    setProcessMessages(messageStatus.Done)
    setHasCode(false)
    setCode("")
    setIsFeedbackOpen(false)
    setCurrentFeedback("")
    setIsFeedbackSent(false)
    setIsDropDownCompleted(false)
    
  }

  const stopGenerating = () => {
    abortFuncs.current.forEach(a => a.abort())
    setShowLoadingMessage(false)
    setIsLoading(false)
  }

  useEffect(() => {
    if (appStateContext?.state.currentChat) {
      setMessages(appStateContext.state.currentChat.messages)
    } else {
      setMessages([])
    }
  }, [appStateContext?.state.currentChat])

  useLayoutEffect(() => {
    const saveToDB = async (messages: ChatMessage[], id: string) => {
      const response = await historyUpdate(messages, id)
      return response
    }

    if (appStateContext && appStateContext.state.currentChat && processMessages === messageStatus.Done) {
      if (appStateContext.state.isCosmosDBAvailable.cosmosDB) {
        if (!appStateContext?.state.currentChat?.messages) {
          console.error('Failure fetching current chat state.')
          return
        }
        const noContentError = appStateContext.state.currentChat.messages.find(m => m.role === ERROR)

        if (!noContentError?.content.includes(NO_CONTENT_ERROR)) {
          saveToDB(appStateContext.state.currentChat.messages, appStateContext.state.currentChat.id)
            .then(res => {
              if (!res.ok) {
                let errorMessage =
                'Ha habido un error. No se pueden guardar respuestas. Porfavor intente de nuevo. Si el problema persiste, porfavor contacte con el administrador.'
                let errorChatMsg: ChatMessage = {
                  id: uuid(),
                  role: ERROR,
                  content: errorMessage,
                  date: new Date().toISOString()
                }
                if (!appStateContext?.state.currentChat?.messages) {
                  let err: Error = {
                    ...new Error(),
                    message: 'Error recogiendo el estado del chat.'
                  }
                  throw err
                }
                setMessages([...appStateContext?.state.currentChat?.messages, errorChatMsg])
              }
              return res as Response
            })
            .catch(err => {
              console.error('Error: ', err)
              let errRes: Response = {
                ...new Response(),
                ok: false,
                status: 500
              }
              return errRes
            })
        }
      } else {
      }
      appStateContext?.dispatch({ type: 'UPDATE_CHAT_HISTORY', payload: appStateContext.state.currentChat })
      setMessages(appStateContext.state.currentChat.messages)
      setProcessMessages(messageStatus.NotRunning)
    }
  }, [processMessages])

  useEffect(() => {
    if (AUTH_ENABLED !== undefined) getUserInfoList()
  }, [AUTH_ENABLED])

  useLayoutEffect(() => {
    chatMessageStreamEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [showLoadingMessage, processMessages])

  const onShowCitation = (citation: Citation) => {
    setActiveCitation(citation)
    setIsCitationPanelOpen(true)
  }

  const onShowExecResult = () => {
    setIsIntentsPanelOpen(true)
  }

  const onViewSource = (citation: Citation) => {
    if (citation.url && !citation.url.includes('blob.core')) {
      window.open(citation.url, '_blank')
    }
  }

  const parseCitationFromMessage = (message: ChatMessage) => {
    if (message?.role && message?.role === 'tool') {
      try {
        const toolMessage = JSON.parse(message.content) as ToolMessageContent
        return toolMessage.citations
      } catch {
        return []
      }
    }
    return []
  }

  const parsePlotFromMessage = (message: ChatMessage) => {
    if (message?.role && message?.role === "tool") {
      try {
        const execResults = JSON.parse(message.content) as AzureSqlServerExecResults;
        const codeExecResult = execResults.all_exec_results.at(-1)?.code_exec_result;
        if (codeExecResult === undefined) {
          return null;
        }
        return codeExecResult;
      }
      catch {
        return null;
      }
      // const execResults = JSON.parse(message.content) as AzureSqlServerExecResults;
      // return execResults.all_exec_results.at(-1)?.code_exec_result;
    }
    return null;
  }

  const disabledButton = () => {
    return (
      isLoading ||
      (messages && messages.length === 0) ||
      clearingChat ||
      appStateContext?.state.chatHistoryLoadingState === ChatHistoryLoadingState.Loading
    )
  }

  return (
    <div className={styles.container} role="main">
      {showAuthMessage ? (
        <Stack className={styles.chatEmptyState}>
          <ShieldLockRegular
            className={styles.chatIcon}
            style={{ color: 'darkorange', height: '200px', width: '200px' }}
          />
          <h1 className={styles.chatEmptyStateTitle}>Authentication Not Configured</h1>
          <h2 className={styles.chatEmptyStateSubtitle}>
            This app does not have authentication configured. Please add an identity provider by finding your app in the{' '}
            <a href="https://portal.azure.com/" target="_blank">
              Azure Portal
            </a>
            and following{' '}
            <a
              href="https://learn.microsoft.com/en-us/azure/app-service/scenario-secure-app-authentication-app-service#3-configure-authentication-and-authorization"
              target="_blank">
              these instructions
            </a>
            .
          </h2>
          <h2 className={styles.chatEmptyStateSubtitle} style={{ fontSize: '20px' }}>
            <strong>Authentication configuration takes a few minutes to apply. </strong>
          </h2>
          <h2 className={styles.chatEmptyStateSubtitle} style={{ fontSize: '20px' }}>
            <strong>If you deployed in the last 10 minutes, please wait and reload the page after 10 minutes.</strong>
          </h2>
        </Stack>
      ) : (
        <Stack horizontal className={styles.chatRoot}>
          <div className={styles.chatContainer}>
          {((!messages && !oldMessages) || (messages.length < 1 && oldMessages.length < 1)) ? (
                <Stack className={styles.chatEmptyState}>
                  <img src={ui?.chat_logo ? ui.chat_logo : Contoso} className={styles.chatIcon} aria-hidden="true" />
                  <h1 className={styles.chatEmptyStateTitle}>{ui?.chat_title}</h1>
                  <h2 className={styles.chatEmptyStateSubtitle}>{ui?.chat_description}</h2>
                </Stack>
              ) : (
                <div className={styles.chatMessageStream} style={{ marginBottom: isLoading ? '40px' : '0px' }} role="log">
                  {(oldMessages.concat(messages)).map((answer, index) => (
                    <>
                      {answer.role === 'user' ? (
                        <div className={styles.chatMessageUser} tabIndex={0}>
                          <div className={styles.chatMessageUserMessage}>{answer.content}</div>
                        </div>
                      ) : answer.role === 'assistant' ? (
                        <div className={styles.chatMessageGpt}>
                          <Answer
                            answer={{
                              answer: answer.content,
                              citations: parseCitationFromMessage(oldMessages.concat(messages)[index - 1]),
                              plotly_data: parsePlotFromMessage(oldMessages.concat(messages)[index - 1]),
                              message_id: answer.id,
                              feedback: answer.feedback,
                              exec_results: execResults
                            }}
                            onCitationClicked={c => onShowCitation(c)}
                            onExectResultClicked={() => onShowExecResult()}
                          />
                        </div>
                      ) : answer.role === ERROR ? (
                        <div className={styles.chatMessageError}>
                          <Stack horizontal className={styles.chatMessageErrorContent}>
                            <ErrorCircleRegular className={styles.errorIcon} style={{ color: 'rgba(182, 52, 67, 1)' }} />
                            <span>Error</span>
                          </Stack>
                          <span className={styles.chatMessageErrorContent}>{answer.content}</span>
                        </div>
                      ) : null}
                    </>
                  ))}
                  {showLoadingMessage && (
                    <>
                      <div className={styles.chatMessageGpt}>
                        <Answer
                          answer={{
                            answer: "Generando respuesta...",
                            citations: [],
                            plotly_data: null
                          }}
                          onCitationClicked={() => null}
                          onExectResultClicked={() => null}
                        />
                      </div>
                    </>
                  )}
                  <div ref={chatMessageStreamEnd} />
                </div>
            )}
            <Stack className={styles.chatInput}>
              <Stack horizontal className={styles.chatInput}>
                {isLoading && (
                  <Stack
                    horizontal
                    title="Detener la generación"
                    className={styles.stopGeneratingContainer}
                    role="button"
                    aria-label="Stop generating"
                    tabIndex={0}
                    onClick={stopGenerating}
                    onKeyDown={e => (e.key === 'Enter' || e.key === ' ' ? stopGenerating() : null)}>
                    <SquareRegular className={styles.stopGeneratingIcon} aria-hidden="true" />
                    <span className={styles.stopGeneratingText} aria-hidden="true">
                          Detener
                    </span>
                  </Stack>
                )}
                <Stack>
                    {appStateContext?.state.isCosmosDBAvailable?.status !== CosmosDBStatus.NotConfigured && (
                      <CommandBarButton
                        role="button"
                        styles={{
                          icon: {
                            color: '#FFFFFF'
                          },
                          iconDisabled: {
                            color: '#BDBDBD !important'
                          },
                          root: {
                            color: '#FFFFFF',
                            background:
                              'radial-gradient(109.81% 107.82% at 100.1% 90.19%, #E33A3A 33.63%, #C32D2D 90.31%, #0E0101 100%)'
                          },
                          rootDisabled: {
                            background: '#F0F0F0'
                          }
                        }}
                        className={styles.newChatIcon}
                        iconProps={{ iconName: 'Add' }}
                        onClick={newChat}
                        disabled={disabledButton()}
                        title="Empezar nueva consulta"
                        aria-label="start a new chat button"
                      />
                    )}
                    <CommandBarButton
                      role="button"
                      styles={{
                        icon: {
                          color: '#FFFFFF'
                        },
                        iconDisabled: {
                          color: '#BDBDBD !important'
                        },
                        root: {
                          color: '#FFFFFF',
                          background:
                            'radial-gradient(109.81% 107.82% at 100.1% 90.19%, #E33A3A 33.63%, #C32D2D 90.31%, #0E0101 100%)'
                        },
                        rootDisabled: {
                          background: '#F0F0F0'
                        }
                      }}
                      className={
                        appStateContext?.state.isCosmosDBAvailable?.status !== CosmosDBStatus.NotConfigured
                          ? styles.clearChatBroom
                          : styles.clearChatBroomNoCosmos
                      }
                      iconProps={{ iconName: 'Broom' }}
                      onClick={
                        appStateContext?.state.isCosmosDBAvailable?.status !== CosmosDBStatus.NotConfigured
                          ? clearChat
                          : newChat
                      }
                      title="Empezar nueva consulta"
                      disabled={disabledButton()}
                      aria-label="clear chat button"
                    />
                    {hasCode && <CommandBarButton
                    role="button"
                    styles={isFeedbackOpen ?
                      {
                        icon: {
                          color: '#FFFFFF'
                        },
                        iconDisabled: {
                          color: '#BDBDBD !important'
                        },
                        root: {
                          color: '#FFFFFF',
                          background:
                            'radial-gradient(109.81% 107.82% at 100.1% 90.19%, #af0909 90.00%, #af0909 90.31%, #af0909 100%)'
                        },
                        rootDisabled: {
                          background: '#F0F0F0'
                        }
                      } 
                      :{
                      icon: {
                        color: '#FFFFFF'
                      },
                      iconDisabled: {
                        color: '#BDBDBD !important'
                      },
                      root: {
                        color: '#FFFFFF',
                        background:
                          'radial-gradient(109.81% 107.82% at 100.1% 90.19%, #0E0101 90.00%, #0E0101 90.31%, #0E0101 100%)'
                      },
                      rootDisabled: {
                        background: '#F0F0F0'
                      }
                    }}
                    className={
                      styles.feedbackIcon
                    }
                    title="Dar feedback de su experiencia"
                    iconProps={{ iconName: 'Feedback'}}
                    onClick={()=>setIsFeedbackOpen(!isFeedbackOpen)}
                    disabled={false}
                    aria-label="feedback button"
                  />}
                  <Dialog
                    hidden={hideErrorDialog}
                    onDismiss={handleErrorDialogClose}
                    dialogContentProps={errorDialogContentProps}
                    modalProps={modalProps}></Dialog>
                </Stack>
                {!hasCode ?
                (<QuestionInput
                  clearOnSend
                  placeholder="Escriba aquí su código."
                  buttonTitle='Confirmar código de consulta'
                  disabled={isLoading}
                  onSend={(code) => {
                    setCode(code)
                    StoreCode(code)
                    setHasCode(true)
                    setOldMessages([...oldMessages, { id: uuid(),
                      role: 'assistant',
                      content: `SESIÓN EMPEZADA CON CÓDIGO: ${code}`,
                      date: new Date().toISOString()
                    } as ChatMessage])
                  }}
                />)
                : (isFeedbackOpen && !isDropDownCompleted) ?
                (<DropDown categories={["Seleccione aquí su feedback","OK", "OK con matices", "KO", "No Aplica"]} setSelectedCategory={(feedback:string) => {
                  setCurrentFeedback(feedback)
                  setIsDropDownCompleted(true)
                }}/>) : isDropDownCompleted ?
                <QuestionInput
                  canSendEmpty
                  clearOnSend
                  buttonTitle='Enviar feedback'
                  placeholder="Escriba aquí su feedback."
                  disabled={!isFeedbackOpen || isLoading}
                  onSend={(feedback,id) => {
                    setIsFeedbackSent(true)
                    updateFeedback(currentFeedback + "  |  "+feedback,id)
                    newChat()
                  }}
                /> 
                :(<QuestionInput
                  clearOnSend
                  buttonTitle='Enviar consulta'
                  placeholder="Escriba aquí su consulta."
                  disabled={isFeedbackOpen || isLoading}
                  onSend={(question, id) => {
                    appStateContext?.state.isCosmosDBAvailable?.cosmosDB
                      ? makeApiRequestWithCosmosDB(question, id)
                      : makeApiRequestWithoutCosmosDB(question, id)
                  }}
                  conversationId={
                    appStateContext?.state.currentChat?.id ? appStateContext?.state.currentChat?.id : undefined
                  }
                />)}
              </Stack>
            </Stack>
          </div>
          {/* Citation Panel */}
          {!isFeedbackOpen && messages && messages.length >0 && isCitationPanelOpen && activeCitation && (
            <Stack.Item className={styles.citationPanel} tabIndex={0} role="tabpanel" aria-label="Citations Panel">
              <Stack
                aria-label="Citations Panel Header Container"
                horizontal
                className={styles.citationPanelHeaderContainer}
                horizontalAlign="space-between"
                verticalAlign="center">
                <span aria-label="Citations" className={styles.citationPanelHeader}>
                  Referencias
                </span>
                <IconButton
                  title="Cerrar el panel de referencias"
                  iconProps={{ iconName: 'Cancel' }}
                  aria-label="Close citations panel"
                  onClick={() => setIsCitationPanelOpen(false)}
                />
              </Stack>
              <h5
                className={styles.citationPanelTitle}
                tabIndex={0}
                title={activeCitation.url && !activeCitation.url.includes('blob.core')
                    ? activeCitation.url
                    : activeCitation.title ?? ''
                }
                onClick={() => onViewSource(activeCitation)}>
                {activeCitation.title}
              </h5>
              <div tabIndex={0}>
                <ReactMarkdown
                  linkTarget="_blank"
                  className={styles.citationPanelContent}
                  children={DOMPurify.sanitize(activeCitation.content, { ALLOWED_TAGS: XSSAllowTags })}
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                />
              </div>
            </Stack.Item>
          )}
          {!isFeedbackOpen && messages && messages.length > 0 && isIntentsPanelOpen && (
            <Stack.Item className={styles.citationPanel} tabIndex={0} role="tabpanel" aria-label="Exec Results Panel">
              <Stack
                aria-label="Intents Panel Header Container"
                horizontal
                className={styles.citationPanelHeaderContainer}
                horizontalAlign="space-between"
                verticalAlign="center">
                <span aria-label="Intents" className={styles.citationPanelHeader}>
                  Exec Results
                </span>
                <IconButton
                  iconProps={{ iconName: 'Cancel' }}
                  aria-label="Close intents panel"
                  onClick={() => setIsIntentsPanelOpen(false)}
                />
              </Stack>
              <Stack horizontalAlign="space-between">
                {execResults.map((execResult) => {
                  return (
                    <Stack className={styles.exectResultList} verticalAlign="space-between">
                      <p><span>Intent:</span> {execResult.intent}</p>
                      {execResult.search_query && <p><span>Search Query:</span> {execResult.search_query}</p>}
                      {execResult.search_result && <p><span>Search Result:</span> {execResult.search_result}</p>}
                    </Stack>
                  )
                })}
              </Stack>
            </Stack.Item>
          )}
          {appStateContext?.state.isChatHistoryOpen &&
            appStateContext?.state.isCosmosDBAvailable?.status !== CosmosDBStatus.NotConfigured && <ChatHistoryPanel />}
        </Stack>
      )}
    </div>
  )
}

export default Chat
