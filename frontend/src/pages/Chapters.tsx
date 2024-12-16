import { useEffect, useState } from 'react'
import { API_URL } from '../utils/credentials'
import { ChapterDataType } from '../lib/types'
import FontAwesomeIconWrapper from '../lib/FontAwesomeIconWrapper'
import Card from '../components/Card'
import { getFilteredIcons, handleSocialUrls } from '../lib/utils'

export default function Chapters() {
  const [chapterData, setChapterData] = useState<ChapterDataType | null>(null)

    useEffect(() => {
      document.title = 'OWASP Chapters'
      const fetchApiData = async () => {
        try {
          const response = await fetch(`${API_URL}/owasp/search/chapter`)
          const data = await response.json()
          setChapterData(data)
        } catch (error) {
          console.error(error)
        }
      }
      fetchApiData()
    }, [])

  return (
    <div className="flex min-h-screen w-full flex-col items-center justify-normal p-5 text-text md:p-20">
      <div className="flex h-fit w-full flex-col items-center justify-normal gap-4">
        {chapterData &&
          chapterData.chapters.map((chapter, index) => {
            const params: string[] = ['idx_updated_at']
            const filteredIcons = getFilteredIcons(chapter, params)
            const formattedUrls = handleSocialUrls(chapter.idx_related_urls)

              const SubmitButton = {
                label: 'Learn More',
                icon: <FontAwesomeIconWrapper icon="fa-solid fa-people-group" />,
                url: chapter.idx_url,
              }

            return (
              <Card
                key={chapter.objectID || `committee-${index}`}
                title={chapter.idx_name}
                url={chapter.idx_url}
                summary={chapter.idx_summary}
                icons={filteredIcons}
                leaders={chapter.idx_leaders}
                topContributors={chapter.idx_top_contributors}
                button={SubmitButton}
                social={formattedUrls}
                tooltipLabel={`Learn more about ${chapter.idx_name}`}
              />
            )
          })}
      </div>
    </div>
  )
}