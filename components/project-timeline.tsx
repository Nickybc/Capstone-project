import { CheckCircle, Circle, Calendar, ArrowRight } from "lucide-react"

export function ProjectTimeline() {
  return (
    <div className="space-y-8">
      {/* Project Proposal */}
      <div className="flex">
        <div className="flex flex-col items-center">
          <CheckCircle className="h-8 w-8 text-green-500" />
          <div className="h-full w-px bg-border" />
        </div>
        <div className="ml-4 space-y-1 pb-8">
          <div className="flex items-center">
            <Calendar className="mr-2 h-4 w-4 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">March 10, 2025</p>
          </div>
          <h3 className="text-xl font-semibold">Detailed Project Proposal</h3>
          <p className="text-gray-500 dark:text-gray-400">
            Submission of comprehensive project proposal outlining objectives, methodology, and expected outcomes.
          </p>
          <div className="mt-2 flex items-center text-sm text-amber-600">
            <ArrowRight className="mr-1 h-4 w-4" />
            <span>Mentor Assessment: March 20, 2025</span>
          </div>
        </div>
      </div>

      {/* Project Progress Updates 1 */}
      <div className="flex">
        <div className="flex flex-col items-center">
          <CheckCircle className="h-8 w-8 text-green-500" />
          <div className="h-full w-px bg-border" />
        </div>
        <div className="ml-4 space-y-1 pb-8">
          <div className="flex items-center">
            <Calendar className="mr-2 h-4 w-4 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">April 7, 2025</p>
          </div>
          <h3 className="text-xl font-semibold">Project Progress Updates 1</h3>
          <p className="text-gray-500 dark:text-gray-400">
            First progress report detailing initial data collection and feature engineering work.
          </p>
          <div className="mt-2 flex items-center text-sm text-amber-600">
            <ArrowRight className="mr-1 h-4 w-4" />
            <span>Mentor Assessment: April 17, 2025</span>
          </div>
        </div>
      </div>

      {/* Project Progress Updates 2 */}
      <div className="flex">
        <div className="flex flex-col items-center">
          <Circle className="h-8 w-8 text-yellow-500" />
          <div className="h-full w-px bg-border" />
        </div>
        <div className="ml-4 space-y-1 pb-8">
          <div className="flex items-center">
            <Calendar className="mr-2 h-4 w-4 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">May 5, 2025</p>
          </div>
          <h3 className="text-xl font-semibold">Project Progress Updates 2</h3>
          <p className="text-gray-500 dark:text-gray-400">
            Second progress report focusing on model development and initial testing results.
          </p>
          <div className="mt-2 flex items-center text-sm text-amber-600">
            <ArrowRight className="mr-1 h-4 w-4" />
            <span>Mentor Assessment: May 15, 2025</span>
          </div>
        </div>
      </div>

      {/* Interim Report and Presentation */}
      <div className="flex">
        <div className="flex flex-col items-center">
          <Circle className="h-8 w-8 text-gray-300 dark:text-gray-600" />
          <div className="h-full w-px bg-border" />
        </div>
        <div className="ml-4 space-y-1 pb-8">
          <div className="flex items-center">
            <Calendar className="mr-2 h-4 w-4 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">June 1, 2025</p>
          </div>
          <h3 className="text-xl font-semibold">Interim Report and Presentation</h3>
          <p className="text-gray-500 dark:text-gray-400">
            Comprehensive interim report and presentation of project progress, challenges, and preliminary findings.
          </p>
          <div className="mt-2 flex items-center text-sm text-amber-600">
            <ArrowRight className="mr-1 h-4 w-4" />
            <span>Mentor Assessment: June 10, 2025</span>
          </div>
        </div>
      </div>

      {/* Project Progress Updates 3 */}
      <div className="flex">
        <div className="flex flex-col items-center">
          <Circle className="h-8 w-8 text-gray-300 dark:text-gray-600" />
          <div className="h-full w-px bg-border" />
        </div>
        <div className="ml-4 space-y-1 pb-8">
          <div className="flex items-center">
            <Calendar className="mr-2 h-4 w-4 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">June 16, 2025</p>
          </div>
          <h3 className="text-xl font-semibold">Project Progress Updates 3</h3>
          <p className="text-gray-500 dark:text-gray-400">
            Third progress report detailing model refinement, validation results, and implementation progress.
          </p>
          <div className="mt-2 flex items-center text-sm text-amber-600">
            <ArrowRight className="mr-1 h-4 w-4" />
            <span>Mentor Assessment: June 26, 2025</span>
          </div>
        </div>
      </div>

      {/* Project Progress Updates 4 */}
      <div className="flex">
        <div className="flex flex-col items-center">
          <Circle className="h-8 w-8 text-gray-300 dark:text-gray-600" />
          <div className="h-full w-px bg-border" />
        </div>
        <div className="ml-4 space-y-1 pb-8">
          <div className="flex items-center">
            <Calendar className="mr-2 h-4 w-4 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">July 7, 2025</p>
          </div>
          <h3 className="text-xl font-semibold">Project Progress Updates 4</h3>
          <p className="text-gray-500 dark:text-gray-400">
            Final progress report before project completion, focusing on deployment and final evaluations.
          </p>
          <div className="mt-2 flex items-center text-sm text-amber-600">
            <ArrowRight className="mr-1 h-4 w-4" />
            <span>Mentor Assessment: July 17, 2025</span>
          </div>
        </div>
      </div>

      {/* Project Webpage */}
      <div className="flex">
        <div className="flex flex-col items-center">
          <Circle className="h-8 w-8 text-gray-300 dark:text-gray-600" />
        </div>
        <div className="ml-4 space-y-1">
          <div className="flex items-center">
            <Calendar className="mr-2 h-4 w-4 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">July 15, 2025</p>
          </div>
          <h3 className="text-xl font-semibold">Project Webpage</h3>
          <p className="text-gray-500 dark:text-gray-400">
            Completion and submission of final project webpage showcasing the complete project.
          </p>
          <div className="mt-2 flex items-center text-sm text-amber-600">
            <ArrowRight className="mr-1 h-4 w-4" />
            <span>Final Assessment: July 25, 2025</span>
          </div>
        </div>
      </div>
    </div>
  )
}
