import {
  Accordion,
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
  Badge,
  Button,
  Card,
  CardBody,
  CircularProgress,
  FormControl,
  Heading,
  Icon,
  IconButton,
  Input,
  Text,
} from '@chakra-ui/react'
import { json } from '@codemirror/lang-json'
import Form from '@rjsf/chakra-ui'
import validator from '@rjsf/validator-ajv8'
import CodeMirror from '@uiw/react-codemirror'
import Ajv from 'ajv'
import React from 'react'
import { useNavigate } from 'react-router-dom'
import { suggestExtractor, useCreateExtractor } from '../api'

import { ChatBubbleBottomCenterTextIcon } from '@heroicons/react/24/outline'
import { useMutation } from '@tanstack/react-query'

const ArrowUpIconImported = (props) => {
  return <Icon as={ChatBubbleBottomCenterTextIcon} {...props} />
}

const ajc = new Ajv()

/**
 * Component to create a new extractor with fields for name, description, schema, and examples
 */
const CreateExtractor = ({}) => {
  const startSchema = '{}'
  // You might use a mutation hook here if you're using something like React Query for state management
  const [schema, setSchema] = React.useState(startSchema)
  const [lastValidSchema, setLastValidSchema] = React.useState(
    JSON.parse(startSchema)
  )
  const [currentSchemaValid, setCurrentSchemaValid] = React.useState(true)
  const [userInput, setUserInput] = React.useState('')

  const suggestMutation = useMutation({
    mutationFn: suggestExtractor,
    onSuccess: (data) => {
      let prettySchema = data.json_schema

      try {
        prettySchema = JSON.stringify(JSON.parse(data.json_schema), null, 2)
      } catch (e) {}

      setSchema(prettySchema)
    },
  })

  const navigate = useNavigate()
  const { mutate, isLoading } = useCreateExtractor({
    onSuccess: (data) => {
      navigate(`/e/${data.uuid}`)
    },
  })

  React.useMemo(() => {
    try {
      const parsedSchema = JSON.parse(schema)
      ajc.compile(parsedSchema)
      setCurrentSchemaValid(true)
      setLastValidSchema(parsedSchema)
    } catch (e) {
      setCurrentSchemaValid(false)
    }
  }, [schema])

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const instruction = ''
    const objectSchema = JSON.parse(schema)
    // Extract information from schema like name, and description
    const name = objectSchema.title || 'Unnamed'
    const description = objectSchema.description || ''
    // backend uses varchar(100) for description
    const shortDescription =
      description.length > 100
        ? description.substring(0, 95) + '...'
        : description

    mutate({
      name,
      description: shortDescription,
      schema: objectSchema,
      instruction,
    })
  }

  const handleSuggest = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const description = event.currentTarget.userInput.value
    if (description === '') {
      return
    }
    console.log(
      `Making request with description: ${description} and schema: ${schema}`
    )
    suggestMutation.mutate({ description, jsonSchema: schema })
    setUserInput('')
  }

  return (
    <div className="w-4/5 m-auto">
      <Heading size={'md'} className="m-auto w-4/5" textAlign={'center'}>
        What would you like to extract today?
      </Heading>
      <form className="m-auto flex flex gap-2 mt-5" onSubmit={handleSuggest}>
        <FormControl id="userInput">
          <Input
            htmlSize={4}
            width="100%"
            autoFocus
            height="auto"
            placeholder="Describe your extraction task..."
            value={userInput}
            onChange={(event) => setUserInput(event.target.value)}
          ></Input>
        </FormControl>
        {suggestMutation.isPending ? (
          <CircularProgress isIndeterminate />
        ) : (
          <IconButton
            type="submit"
            icon={<ArrowUpIconImported />}
            aria-label="OK"
            colorScheme="blue"
          />
        )}
      </form>
      <form
        className="m-auto flex flex-col content-between gap-5 mt-10"
        onSubmit={handleSubmit}
      >
        <div className="divider">OR</div>
        <Accordion allowToggle={true}>
          <AccordionItem>
            <AccordionButton>
              Edit JSON Schema
              <div className="ml-auto">
                {currentSchemaValid ? (
                  <Badge colorScheme="green">OK</Badge>
                ) : (
                  <Badge colorScheme="red">Errors!</Badge>
                )}
                <AccordionIcon />
              </div>
            </AccordionButton>
            <AccordionPanel>
              <FormControl isInvalid={!currentSchemaValid}>
                <CodeMirror
                  id="schema"
                  value={schema}
                  aria-label="JSON Schema Editor"
                  onChange={(value) => setSchema(value)}
                  basicSetup={{ autocompletion: true }}
                  extensions={[json()]}
                  minHeight="300px"
                  className="border-4 border-slate-300 border-double"
                />
              </FormControl>
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
        {Object.keys(lastValidSchema).length !== 0 && (
          <>
            <Heading size="md">Preview</Heading>
            {!currentSchemaValid && (
              <Text color="red.500">
                JSON Schema has errors. Showing previous valid JSON Schema.
              </Text>
            )}
            <Card>
              <CardBody>
                <Form
                  schema={lastValidSchema}
                  validator={validator}
                  disabled={!currentSchemaValid || suggestMutation.isPending}
                  children={true} // Hide the submit button
                />
              </CardBody>
            </Card>
          </>
        )}
        <Button className="btn" type="submit" size="lg">
          Create
        </Button>
      </form>
    </div>
  )
}

export default CreateExtractor
